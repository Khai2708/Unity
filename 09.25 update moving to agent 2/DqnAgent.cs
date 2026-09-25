using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class DqnAgent : MonoBehaviour
{
    [Header("Environment References")]
    public FactoryEnvManager factoryEnv;
    public RepairTaskManager repairTaskManager;

    [Header("TCP")]
    public DqnTcpClient tcpClient;
    public bool sendTransitionOverTcp = false;

    [Header("Debug")]
    public bool debugLogs = true;
    public bool logStateVector = true;
    public int maxStateElementsToLog = 48;

    [Header("Grid Parameters")]
    private const int GRID_ROWS = 3;
    private const int GRID_COLS = 5;
    private const int NUM_CHANNELS = 6; // 6 features per cell (removed capacity + occupancy — both constant in this topology)
    private const int CELL_FEATURE_DIM = GRID_ROWS * GRID_COLS * NUM_CHANNELS; // 120

    [Header("History (t, t-3, t-5)")]
    public int historyLength = 6; // must be >= lagStepsB + 1
    public int snapshotIntervalTU = 1;
    public int lagStepsA = 3; // how many snapshots back for the "t-3" slot
    public int lagStepsB = 5; // how many snapshots back for the "t-5" slot
    [Header("Action Request Timing")]
    public int waitFramesBeforeActionRequest = 1;
    public bool waitUntilFaultVisible = true;
    public int maxRequestSyncFrames = 2;

    [Header("Normalisation")]
    public float maxQueueCapacity = 50f;
    public float maxQueueCount = 50f;
    public float maxStateDuration = 100f;

    private List<float[]> snapshotHistory = new List<float[]>();   // each = 120 floats
    private List<float[]> spawnerHistory = new List<float[]>();    // each = numSpawners
    private int nextSnapshotTU = 0;
    private int lastSnapshotTU = int.MinValue;

    private Dictionary<int, (int prevState, int duration)> nodeStateDuration = new Dictionary<int, (int, int)>();


    private bool waitingActionReply = false;
    private bool hasLastActionReply = false;
    private int lastChosenNodeId = -1;
    private int[] lastCandidateNodeIds = null;
    private float[] lastQValues = null;
    private float lastEpsilon = 0f;
    private bool lastIsRandom = false;

    private int numSpawners;

    [Serializable]
    public class ActionRequestMessage
    {
        public string type = "action_request";
        public float[] state;               // 369 floats (grid 360 + spawner 9)
        public int[] candidate_node_ids;
        public float epsilon;
    }

    [Serializable]
    public class TransitionMessage
    {
        public string type = "transition";
        public int action_id;
        public int node_id;
        public float reward;
        public float qd_raw_delta;
        public float qd_norm;
        public float qd_t;
        public float qd_n;
        public float ar_t;
        public float ec_t;
        public float is_fault_hit;
        public float[] state_t;             // 369
        public float[] state_tp1;           // 369
        public int[] next_candidate_node_ids;
        // Throughput fields
        public float sim_time_sec;
        public int total_exits;
    }
    private class PendingTransition
    {
        public float[] state;
        public int actionId;
        public float isFaultHit;
    }
    private Dictionary<int, PendingTransition> pendingTransitions = new Dictionary<int, PendingTransition>();
    private int transitionStepCounter = 0;

    public void SetSelectionOutcome(int nodeId, bool isFaultHit)
    {
        if (pendingTransitions.TryGetValue(nodeId, out var pt))
            pt.isFaultHit = isFaultHit ? 1f : 0f;
    }

    void Start()
    {
        if (factoryEnv != null && factoryEnv.spawners != null)
            numSpawners = factoryEnv.spawners.Length;
        else
            numSpawners = 0;

        historyLength = Mathf.Max(historyLength, lagStepsB + 1);
        ForceRefreshSnapshotsNow();
    }

    void Awake()
    {
        if (factoryEnv == null)
            factoryEnv = FactoryEnvManager.Instance;

        if (tcpClient != null)
            tcpClient.OnActionReply += HandleActionReplyFromPython;

        historyLength = Mathf.Max(historyLength, lagStepsB + 1);
        snapshotIntervalTU = Mathf.Max(1, snapshotIntervalTU);
        nextSnapshotTU = GetCurrentSimulationTU() + snapshotIntervalTU;
    }

    void OnDestroy()
    {
        if (tcpClient != null)
            tcpClient.OnActionReply -= HandleActionReplyFromPython;
    }

    void Update()
    {
        int currentTU = GetCurrentSimulationTU();
        if (currentTU >= nextSnapshotTU)
        {
            ForceRefreshSnapshotsNow();
            nextSnapshotTU = currentTU + snapshotIntervalTU;
        }
    }

    int GetCurrentSimulationTU()
    {
        return factoryEnv != null ? factoryEnv.GetCurrentTU() : 0;
    }

    float GetSimulationTimeSeconds()
    {
        return factoryEnv != null ? factoryEnv.GetSimulationTimeSeconds() : Time.time;
    }

    public void SyncFactoryStateNow()
    {
        if (factoryEnv == null) return;

        MethodInfo syncMethod = factoryEnv.GetType().GetMethod(
            "SyncNodeStatesImmediate",
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic
        );
        if (syncMethod != null)
            syncMethod.Invoke(factoryEnv, null);
        else
            factoryEnv.SendMessage("UpdateNodeStates", SendMessageOptions.DontRequireReceiver);
    }

    public void ForceRefreshSnapshotsNow()
    {
        SyncFactoryStateNow();
        TakeSnapshot();
    }

    void TakeSnapshot()
    {
        float[] currentGrid = BuildCurrentGridSnapshot();   // 120 floats
        float[] currentRates = factoryEnv != null ? factoryEnv.GetCurrentSpawnerRates() : Array.Empty<float>();
        int currentTU = GetCurrentSimulationTU();

        if (snapshotHistory.Count > 0 && lastSnapshotTU == currentTU)
        {
            snapshotHistory[snapshotHistory.Count - 1] = currentGrid;
            if (spawnerHistory.Count > 0 && currentRates.Length > 0)
                spawnerHistory[spawnerHistory.Count - 1] = currentRates;
        }
        else
        {
            snapshotHistory.Add(currentGrid);
            if (currentRates.Length > 0)
                spawnerHistory.Add(currentRates);
            else
                spawnerHistory.Add(new float[numSpawners]); // zeros
            lastSnapshotTU = currentTU;
        }

        while (snapshotHistory.Count > historyLength)
        {
            snapshotHistory.RemoveAt(0);
            if (spawnerHistory.Count > 0)
                spawnerHistory.RemoveAt(0);
        }

        if (debugLogs)
            Debug.Log($"[DqnAgent] Snapshot taken. Grid history={snapshotHistory.Count}, Spawner history={spawnerHistory.Count}");
    }

    float[] BuildCurrentGridSnapshot()
    {
        float[] flat = new float[CELL_FEATURE_DIM]; // 120 zeros

        if (factoryEnv == null || factoryEnv.Nodes == null)
            return flat;

        var nodesDict = factoryEnv.Nodes;
        List<FactoryEnvManager.NodeData> tunnelNodes = new List<FactoryEnvManager.NodeData>();
        foreach (var kv in nodesDict)
        {
            if (!kv.Value.isSpawner && kv.Value.tunnel != null)
                tunnelNodes.Add(kv.Value);
        }
        tunnelNodes.Sort((a, b) => a.nodeId.CompareTo(b.nodeId));

        float[,,] cellSum = new float[GRID_ROWS, GRID_COLS, NUM_CHANNELS];
        int[,] cellCount = new int[GRID_ROWS, GRID_COLS];

        foreach (var node in tunnelNodes)
        {
            TunnelController tunnel = node.tunnel;
            int row = tunnel.lineIndex;
            int col = tunnel.positionIndex;
            row = Mathf.Clamp(row, 0, GRID_ROWS - 1);
            col = Mathf.Clamp(col, 0, GRID_COLS - 1);

            int stateIdx = 0;
            switch (node.tunnelState)
            {
                case TunnelController.TunnelState.HALF_HOLD: stateIdx = 1; break;
                case TunnelController.TunnelState.HOLD:     stateIdx = 2; break;
                case TunnelController.TunnelState.FAULT:    stateIdx = 3; break;
                default: stateIdx = 0; break;
            }

            int duration = UpdateNodeDuration(node.nodeId, stateIdx);

            float normCount = Mathf.Clamp01(node.queueCount / maxQueueCount);
            float normDuration = Mathf.Clamp01(duration / maxStateDuration);

            cellCount[row, col]++;

            cellSum[row, col, 0] += normCount;
            cellSum[row, col, 1] += (stateIdx == 0) ? 1 : 0;
            cellSum[row, col, 2] += (stateIdx == 1) ? 1 : 0;
            cellSum[row, col, 3] += (stateIdx == 2) ? 1 : 0;
            cellSum[row, col, 4] += (stateIdx == 3) ? 1 : 0;
            cellSum[row, col, 5] += normDuration;
        }

        int idx = 0;
        for (int row = 0; row < GRID_ROWS; row++)
        {
            for (int col = 0; col < GRID_COLS; col++)
            {
                int cnt = cellCount[row, col];
                float invCnt = (cnt > 0) ? 1f / cnt : 0f;

                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                    flat[idx++] = cellSum[row, col, ch] * invCnt;
            }
        }
        return flat;
    }

    int UpdateNodeDuration(int nodeId, int currentState)
    {
        if (nodeStateDuration.TryGetValue(nodeId, out var prev))
        {
            if (prev.prevState == currentState)
            {
                int newDuration = prev.duration + 1;
                nodeStateDuration[nodeId] = (currentState, newDuration);
                return newDuration;
            }
            else
            {
                nodeStateDuration[nodeId] = (currentState, 1);
                return 1;
            }
        }
        else
        {
            nodeStateDuration[nodeId] = (currentState, 1);
            return 1;
        }
    }

    float[] CloneStateOrZeros(int index, bool isGrid)
    {
        if (isGrid)
        {
            if (snapshotHistory.Count == 0)
                return new float[CELL_FEATURE_DIM];
            index = Mathf.Clamp(index, 0, snapshotHistory.Count - 1);
            float[] src = snapshotHistory[index];
            float[] dst = new float[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }
        else
        {
            if (spawnerHistory.Count == 0)
                return new float[numSpawners];
            index = Mathf.Clamp(index, 0, spawnerHistory.Count - 1);
            float[] src = spawnerHistory[index];
            float[] dst = new float[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }
    }

    float[] BuildFullState()
    {
        ForceRefreshSnapshotsNow();

        int n = snapshotHistory.Count;
        int nSpawn = spawnerHistory.Count;

        int idxT = n - 1;
        int idxT3 = Mathf.Max(0, n - 1 - lagStepsA);
        int idxT5 = Mathf.Max(0, n - 1 - lagStepsB);

        int idxSpawnT = nSpawn - 1;
        int idxSpawnT3 = Mathf.Max(0, nSpawn - 1 - lagStepsA);
        int idxSpawnT5 = Mathf.Max(0, nSpawn - 1 - lagStepsB);

        float[] gridT = CloneStateOrZeros(idxT, true);
        float[] gridT3 = CloneStateOrZeros(idxT3, true);
        float[] gridT5 = CloneStateOrZeros(idxT5, true);

        float[] spawnT = CloneStateOrZeros(idxSpawnT, false);
        float[] spawnT3 = CloneStateOrZeros(idxSpawnT3, false);
        float[] spawnT5 = CloneStateOrZeros(idxSpawnT5, false);

        // concatenate: gridT + gridT3 + gridT5 + spawnT + spawnT3 + spawnT5
        int gridPart = CELL_FEATURE_DIM * 3;
        int spawnPart = numSpawners * 3;
        float[] full = new float[gridPart + spawnPart];

        Array.Copy(gridT, 0, full, 0, CELL_FEATURE_DIM);
        Array.Copy(gridT3, 0, full, CELL_FEATURE_DIM, CELL_FEATURE_DIM);
        Array.Copy(gridT5, 0, full, CELL_FEATURE_DIM * 2, CELL_FEATURE_DIM);

        Array.Copy(spawnT, 0, full, gridPart, numSpawners);
        Array.Copy(spawnT3, 0, full, gridPart + numSpawners, numSpawners);
        Array.Copy(spawnT5, 0, full, gridPart + numSpawners * 2, numSpawners);

        return full;
    }

    float[] GetConcatenatedGrid()
    {
        return BuildFullState();
    }

    bool HasAnyFault(float[] stateVec)
    {
        if (stateVec == null) return false;
        // Fault is channel index 4 (one-hot FAULT) in the new 6-channel layout
        for (int cell = 0; cell < GRID_ROWS * GRID_COLS; cell++)
        {
            int baseIdx = cell * NUM_CHANNELS;
            if (baseIdx + 4 < stateVec.Length && stateVec[baseIdx + 4] > 0.5f)
                return true;
        }
        return false;
    }

    void LogStateVectorIfNeeded(string tag, float[] stateVec)
    {
        if (!debugLogs || !logStateVector || stateVec == null) return;

        int gridStep = CELL_FEATURE_DIM;              // 120 per history step
        int gridPart = gridStep * 3;                  // 360 total (t, t-3, t-5)
        int spawnStep = numSpawners;                   // per history step
        bool hasFullLayout = stateVec.Length >= gridPart + spawnStep * 3;

        if (!hasFullLayout)
        {
            // Fallback: not the expected 369-style layout, just dump flat.
            int show = Mathf.Min(maxStateElementsToLog, stateVec.Length);
            string[] parts = new string[show];
            for (int i = 0; i < show; i++) parts[i] = stateVec[i].ToString("0.###");
            Debug.Log($"[DqnAgent] {tag} state_dim={stateVec.Length}, head=[{string.Join(", ", parts)}]");
            return;
        }

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.Append($"[DqnAgent] {tag} state_dim={stateVec.Length}\n");

        string[] gridLabels = { "t", "t-3", "t-5" };
        for (int step = 0; step < 3; step++)
        {
            int baseIdx = step * gridStep;
            sb.Append($"  grid {gridLabels[step]}: ");
            for (int cell = 0; cell < GRID_ROWS * GRID_COLS; cell++)
            {
                int row = cell / GRID_COLS;
                int col = cell % GRID_COLS;
                int cellBase = baseIdx + cell * NUM_CHANNELS;
                string[] vals = new string[NUM_CHANNELS];
                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                    vals[ch] = stateVec[cellBase + ch].ToString("0.###");
                sb.Append($"({row},{col})[{string.Join(",", vals)}] ");
            }
            sb.Append("\n");
        }

        string[] spawnLabels = { "t", "t-3", "t-5" };
        for (int step = 0; step < 3; step++)
        {
            int baseIdx = gridPart + step * spawnStep;
            string[] vals = new string[spawnStep];
            for (int s = 0; s < spawnStep; s++)
                vals[s] = stateVec[baseIdx + s].ToString("0.###");
            sb.Append($"  spawner {spawnLabels[step]}: [{string.Join(", ", vals)}]\n");
        }

        Debug.Log(sb.ToString());
    }

    public void RecordAction(int actionId, int nodeId, float[] capturedState = null)
    {
        if (factoryEnv == null)
        {
            Debug.LogError("[DqnAgent] FactoryEnvManager reference is missing.");
            return;
        }

        if (pendingTransitions.ContainsKey(nodeId))
        {
            Debug.LogWarning($"[DqnAgent] Overwriting unfinished transition for node={nodeId}. This node's previous transition was never sent.");
        }

        float[] state = capturedState ?? GetConcatenatedGrid();
        pendingTransitions[nodeId] = new PendingTransition
        {
            state = state,
            actionId = actionId,
            isFaultHit = 0f
        };

        factoryEnv.BeginQueueRewardObservation(nodeId);

        if (debugLogs)
            Debug.Log($"[DqnAgent] RecordAction action={actionId}, node={nodeId}, state_dim={state?.Length ?? 0}, activePending={pendingTransitions.Count}, usedCapturedState={capturedState != null}");
        LogStateVectorIfNeeded("RecordAction", state);
    }

    public void FinishStepAndSend(int nodeId)
    {
        if (!pendingTransitions.TryGetValue(nodeId, out var pt)) return;
        if (factoryEnv == null)
        {
            Debug.LogError("[DqnAgent] FactoryEnvManager reference is missing.");
            return;
        }

        pendingTransitions.Remove(nodeId);

        SendTransitionNow(pt.actionId, nodeId, pt.state, pt.isFaultHit);
    }

    void SendTransitionNow(int actionId, int nodeId, float[] state_t, float isFaultHit)
    {
        float qdRaw = 0f;
        float qdNorm = 0f;
        float reward = factoryEnv.ComputeGlobalReward(nodeId, out qdRaw, out qdNorm);
        float arRaw = (factoryEnv != null) ? factoryEnv.GetLastAssemblyReward() : 0f;
        float ecRaw = (factoryEnv != null) ? factoryEnv.GetLastEnergyReward() : 0f;

        if (factoryEnv != null)
            factoryEnv.NotifyDecisionStepRewardFinalized(nodeId);

        float[] nextState = GetConcatenatedGrid();
        int[] nextCandidateNodeIds = GetNextCandidateNodeIds();

        transitionStepCounter++;

        // Get throughput timing: simulation time and total exits after reward computation (repair finished)
        float simTime = factoryEnv.GetSimulationTimeSeconds();
        int totalExits = factoryEnv.TotalExits;

        var msg = new TransitionMessage
        {
            type = "transition",
            action_id = actionId,
            node_id = nodeId,
            reward = reward,
            qd_raw_delta = qdRaw,
            qd_norm = qdNorm,
            qd_t = qdRaw,
            qd_n = qdNorm,
            ar_t = arRaw,
            ec_t = ecRaw,
            is_fault_hit = isFaultHit,
            state_t = state_t,
            state_tp1 = nextState,
            next_candidate_node_ids = nextCandidateNodeIds,
            sim_time_sec = simTime,
            total_exits = totalExits
        };

        string json = JsonUtility.ToJson(msg);

        if (debugLogs)
        {
            Debug.Log($"[DqnAgent] Transition step={transitionStepCounter} action={actionId}, node={nodeId}, reward={reward:F4}, qd={qdRaw:F4}, ar={arRaw:F4}, ec={ecRaw:F4}, is_fault_hit={isFaultHit:F0}, time={simTime:F2}, exits={totalExits}");
        }
        LogStateVectorIfNeeded("Transition state_t", state_t);
        LogStateVectorIfNeeded("Transition state_tp1", nextState);

        if (sendTransitionOverTcp && tcpClient != null)
        {
            try
            {
                tcpClient.SendJsonLine(json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[DqnAgent] TCP send error: {e}");
            }
        }
    }

    int[] GetNextCandidateNodeIds()
    {
        return repairTaskManager != null ? repairTaskManager.GetFaultyCandidateNodeIds() : Array.Empty<int>();
    }

    public IEnumerator CoRequestActionAndPickNode(List<int> candidates, float epsilon, Action<int, bool, float[]> onDone)
    {
        if (tcpClient == null)
        {
            int fallback = (candidates != null && candidates.Count > 0) ? candidates[UnityEngine.Random.Range(0, candidates.Count)] : -1;
            onDone?.Invoke(fallback, true, null);
            yield break;
        }

        if (candidates == null || candidates.Count == 0)
        {
            onDone?.Invoke(-1, true, null);
            yield break;
        }

        int preWaitFrames = Mathf.Max(0, waitFramesBeforeActionRequest);
        for (int i = 0; i < preWaitFrames; i++)
            yield return null;

        float[] stateVec = GetConcatenatedGrid();

        if (waitUntilFaultVisible)
        {
            int retry = 0;
            while (!HasAnyFault(stateVec) && retry < Mathf.Max(0, maxRequestSyncFrames))
            {
                if (debugLogs)
                    Debug.LogWarning($"[DqnAgent] action_request snapshot has no visible fault. retry={retry + 1}/{maxRequestSyncFrames}");
                yield return null;
                stateVec = GetConcatenatedGrid();
                retry++;
            }
        }

        ActionRequestMessage req = new ActionRequestMessage
        {
            type = "action_request",
            state = stateVec,
            candidate_node_ids = candidates.ToArray(),
            epsilon = epsilon
        };

        string json = JsonUtility.ToJson(req);
        if (debugLogs)
            Debug.Log($"[DqnAgent] action_request sent. candidates=[{string.Join(",", candidates)}], hasFault={HasAnyFault(stateVec)}");
        LogStateVectorIfNeeded("ActionRequest", stateVec);

        waitingActionReply = true;
        hasLastActionReply = false;
        tcpClient.SendJsonLine(json);

        float timeout = 2.0f;
        float startTime = GetSimulationTimeSeconds();
        while (waitingActionReply && (GetSimulationTimeSeconds() - startTime) < timeout)
            yield return null;

        if (!hasLastActionReply)
        {
            int fallback = candidates[UnityEngine.Random.Range(0, candidates.Count)];
            onDone?.Invoke(fallback, true, stateVec);
            yield break;
        }

        onDone?.Invoke(lastChosenNodeId, lastIsRandom, stateVec);
    }

    void HandleActionReplyFromPython(int chosenNodeId, int[] candidateNodeIds, float[] qValues, float epsilon, bool isRandom)
    {
        if (!waitingActionReply) return;
        lastChosenNodeId = chosenNodeId;
        lastCandidateNodeIds = candidateNodeIds;
        lastQValues = qValues;
        lastEpsilon = epsilon;
        lastIsRandom = isRandom;
        hasLastActionReply = true;
        waitingActionReply = false;
        if (debugLogs)
            Debug.Log($"[DqnAgent] action_reply: chosen={chosenNodeId}, random={isRandom}");
    }
}