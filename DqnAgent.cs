using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
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
    public int maxStateElementsToLog = 32;
    public bool logStackTraceOnDuplicateRecord = true;

    [Header("Grid Parameters")]
    private const int GRID_SIZE = 24;
    private const int NUM_CHANNELS = 8;          // per snapshot

    [Header("History (N steps)")]
    [Tooltip("Number of recent snapshots to stack as state")]
    public int historyLength = 3;
    [Tooltip("Interval (seconds) between snapshots")]
    public float snapshotInterval = 1.0f;

    [Header("Normalisation")]
    public float maxQueueCapacity = 50f;
    public float maxQueueCount = 50f;
    public float maxStateDuration = 100f;        // maximum expected duration in steps

    // History buffer (circular)
    private Queue<float[]> snapshotHistory = new Queue<float[]>();
    private float nextSnapshotTime;

    // For state duration tracking per node
    private Dictionary<int, (int prevState, int duration)> nodeStateDuration = new Dictionary<int, (int, int)>();

    // Transition data
    float[] lastState;
    int lastActionId = -1;
    int lastNodeId = -1;
    bool hasPendingTransition = false;
    int transitionStepCounter = 0;

    // Action reply handling
    bool waitingActionReply = false;
    bool hasLastActionReply = false;
    int lastChosenNodeId = -1;
    int[] lastCandidateNodeIds = null;
    float[] lastQValues = null;
    float lastEpsilon = 0f;
    bool lastIsRandom = false;

    // At the top with other fields
    private int numSpawners;
    private Queue<float[]> spawnerHistory = new Queue<float[]>();   // stores spawner rate vectors for each snapshot
    private float[] lastSpawnerRates;  // for s_t

    // In Awake() or Start(), get spawner count
    void Start()
    {
        if (factoryEnv != null && factoryEnv.spawners != null)
            numSpawners = factoryEnv.spawners.Length;
        else
            numSpawners = 0; // fallback
    }

    [Serializable]
    public class ActionRequestMessage
    {
        public string type = "action_request";
        public float[] state;               // flattened grid history
        public float[] spawner_rates;        // flattened spawner rates history
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
        public float bt_raw_delta;
        public float pl_raw_delta;
        public float[] state_t;              // grid history at t
        public float[] state_tp1;            // grid history at t+1
        public float[] spawner_rates_t;      // spawner rates history at t
        public float[] spawner_rates_tp1;    // spawner rates history at t+1
    }

    void Awake()
    {
        if (factoryEnv == null)
            factoryEnv = FactoryEnvManager.Instance;

        if (tcpClient != null)
            tcpClient.OnActionReply += HandleActionReplyFromPython;

        nextSnapshotTime = Time.time + snapshotInterval;
    }

    void OnDestroy()
    {
        if (tcpClient != null)
            tcpClient.OnActionReply -= HandleActionReplyFromPython;
    }

    void Update()
    {
        // Take snapshots at regular intervals
        if (Time.time >= nextSnapshotTime)
        {
            TakeSnapshot();
            nextSnapshotTime = Time.time + snapshotInterval;
        }
    }

    // ------------------------------------------------------------
    // Snapshot & History Management
    // ------------------------------------------------------------

    /// <summary>
    /// Captures the current grid state and pushes it into the history buffer.
    /// </summary>
    void TakeSnapshot()
    {
        float[] currentGrid = BuildCurrentGridSnapshot();
        float[] currentRates = factoryEnv.GetCurrentSpawnerRates(); // you need to implement this

        snapshotHistory.Enqueue(currentGrid);
        spawnerHistory.Enqueue(currentRates);

        while (snapshotHistory.Count > historyLength)
        {
            snapshotHistory.Dequeue();
            spawnerHistory.Dequeue();
        }

        if (debugLogs)
            Debug.Log($"[DqnAgent] Snapshot taken. Grid history: {snapshotHistory.Count}, Spawner history: {spawnerHistory.Count}");
    }

    /// <summary>
    /// Builds a single grid snapshot (flattened) using the current node data.
    /// </summary>
    float[] BuildCurrentGridSnapshot()
    {
        if (factoryEnv == null || factoryEnv.Nodes == null)
            return new float[GRID_SIZE * GRID_SIZE * NUM_CHANNELS];

        var nodesDict = factoryEnv.Nodes;

        // Collect tunnel nodes only (skip spawners)
        List<FactoryEnvManager.NodeData> tunnelNodes = new List<FactoryEnvManager.NodeData>();
        foreach (var kv in nodesDict)
        {
            if (!kv.Value.isSpawner && kv.Value.tunnel != null)
                tunnelNodes.Add(kv.Value);
        }

        int n = tunnelNodes.Count;
        if (n == 0)
            return new float[GRID_SIZE * GRID_SIZE * NUM_CHANNELS];

        // Compute bounding box of tunnel positions
        Vector3 min = tunnelNodes[0].tunnel.transform.position;
        Vector3 max = tunnelNodes[0].tunnel.transform.position;
        for (int i = 1; i < n; i++)
        {
            Vector3 pos = tunnelNodes[i].tunnel.transform.position;
            min = Vector3.Min(min, pos);
            max = Vector3.Max(max, pos);
        }
        float rangeX = Mathf.Max(0.1f, max.x - min.x);
        float rangeZ = Mathf.Max(0.1f, max.z - min.z);

        // Accumulators for averaging
        float[,,] cellSum = new float[GRID_SIZE, GRID_SIZE, NUM_CHANNELS];
        int[,] cellCount = new int[GRID_SIZE, GRID_SIZE];

        // Process each node
        foreach (var node in tunnelNodes)
        {
            Vector3 pos = node.tunnel.transform.position;
            int ix = Mathf.FloorToInt((pos.x - min.x) / rangeX * (GRID_SIZE - 1));
            int iz = Mathf.FloorToInt((pos.z - min.z) / rangeZ * (GRID_SIZE - 1));
            ix = Mathf.Clamp(ix, 0, GRID_SIZE - 1);
            iz = Mathf.Clamp(iz, 0, GRID_SIZE - 1);

            // Node state as integer
            int stateIdx = 0;
            switch (node.tunnelState)
            {
                case TunnelController.TunnelState.HALF_HOLD: stateIdx = 1; break;
                case TunnelController.TunnelState.HOLD:     stateIdx = 2; break;
                case TunnelController.TunnelState.FAULT:    stateIdx = 3; break;
                default: stateIdx = 0; break;
            }

            // Update duration tracking
            int duration = UpdateNodeDuration(node.nodeId, stateIdx);

            // Normalised features
            float normCapacity = Mathf.Clamp01(node.queueCapacity / maxQueueCapacity);
            float normCount = Mathf.Clamp01(node.queueCount / maxQueueCount);
            float normDuration = Mathf.Clamp01(duration / maxStateDuration);

            // Accumulate
            cellCount[ix, iz]++;

            cellSum[ix, iz, 0] += normCapacity;
            cellSum[ix, iz, 1] += normCount;
            // One-hot state
            cellSum[ix, iz, 2] += (stateIdx == 0) ? 1 : 0;
            cellSum[ix, iz, 3] += (stateIdx == 1) ? 1 : 0;
            cellSum[ix, iz, 4] += (stateIdx == 2) ? 1 : 0;
            cellSum[ix, iz, 5] += (stateIdx == 3) ? 1 : 0;
            cellSum[ix, iz, 6] += normDuration;
            // Channel 7 will be existence flag (set later)
        }

        // Flatten the grid (row-major: rows first, then columns, then channels)
        float[] flat = new float[GRID_SIZE * GRID_SIZE * NUM_CHANNELS];
        int idx = 0;
        for (int iz = 0; iz < GRID_SIZE; iz++)
        {
            for (int ix = 0; ix < GRID_SIZE; ix++)
            {
                int cnt = cellCount[ix, iz];
                float invCnt = (cnt > 0) ? 1f / cnt : 0f;

                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                {
                    if (ch == 7)
                        flat[idx++] = (cnt > 0) ? 1f : 0f;   // existence flag
                    else
                        flat[idx++] = cellSum[ix, iz, ch] * invCnt; // average
                }
            }
        }
        return flat;
    }

    /// <summary>
    /// Updates and returns the duration (in steps) that a node has been in its current state.
    /// </summary>
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

    /// <summary>
    /// Returns the concatenated history (last N snapshots) as a single flat array.
    /// If history not yet full, repeats the oldest snapshot or pads with zeros.
    /// </summary>
    float[] GetConcatenatedGrid()
    {
        int snapshotLength = GRID_SIZE * GRID_SIZE * NUM_CHANNELS;
        int totalLength = historyLength * snapshotLength;
        float[] result = new float[totalLength];
        int offset = 0;
        foreach (var snap in snapshotHistory)
        {
            Array.Copy(snap, 0, result, offset, snapshotLength);
            offset += snapshotLength;
        }
        // Pad if needed (same as before)
        if (snapshotHistory.Count < historyLength)
        {
            float[] padSnap = snapshotHistory.Count > 0 ? snapshotHistory.Peek() : new float[snapshotLength];
            while (offset < totalLength)
            {
                Array.Copy(padSnap, 0, result, offset, Math.Min(snapshotLength, totalLength - offset));
                offset += snapshotLength;
            }
        }
        return result;
    }

    float[] GetConcatenatedSpawnerRates()
    {
        int rateLength = numSpawners;
        int totalLength = historyLength * rateLength;
        float[] result = new float[totalLength];
        int offset = 0;
        foreach (var rates in spawnerHistory)
        {
            Array.Copy(rates, 0, result, offset, rateLength);
            offset += rateLength;
        }
        if (spawnerHistory.Count < historyLength)
        {
            float[] padRates = spawnerHistory.Count > 0 ? spawnerHistory.Peek() : new float[rateLength];
            while (offset < totalLength)
            {
                Array.Copy(padRates, 0, result, offset, Math.Min(rateLength, totalLength - offset));
                offset += rateLength;
            }
        }
        return result;
    }

    // ------------------------------------------------------------
    // Action Recording (same as before, but using concatenated state)
    // ------------------------------------------------------------

    public void RecordAction(int actionId, int nodeId)
    {
        if (factoryEnv == null)
        {
            Debug.LogError("[DqnAgent] FactoryEnvManager 참조가 없습니다.");
            return;
        }

        if (hasPendingTransition)
        {
            Debug.LogWarning($"[DqnAgent] RecordAction called before previous transition finished. (prev={lastActionId},{lastNodeId} new={actionId},{nodeId})");
            if (logStackTraceOnDuplicateRecord)
                Debug.Log(new StackTrace(true));
        }

        lastState = GetConcatenatedGrid();
        lastSpawnerRates = GetConcatenatedSpawnerRates();
        lastActionId = actionId;
        lastNodeId = nodeId;
        hasPendingTransition = true;

        if (debugLogs)
            Debug.Log($"[DqnAgent] RecordAction: actionId={actionId}, nodeId={nodeId}, state_dim={lastState.Length}");
    }

    public void FinishStepAndSend()
    {
        if (!hasPendingTransition) return;
        if (factoryEnv == null)
        {
            Debug.LogError("[DqnAgent] FactoryEnvManager 참조가 없습니다.");
            return;
        }

        int actionId = lastActionId;
        int nodeId = lastNodeId;
        float[] state_t = lastState;
        hasPendingTransition = false;

        if (factoryEnv.useObservationWindow)
        {
            factoryEnv.BeginRewardObservation();
            if (debugLogs)
                Debug.Log($"[DqnAgent] BeginRewardObservation() for action {actionId}");
        }

        StartCoroutine(CoWaitWindowAndSend(actionId, nodeId, state_t));
    }

    IEnumerator CoWaitWindowAndSend(int actionId, int nodeId, float[] state_t)
    {
        float waitT = factoryEnv.useObservationWindow ? Mathf.Max(0f, factoryEnv.observationWindow) : 0f;
        if (waitT > 0f)
        {
            yield return new WaitForSeconds(waitT);
        }
        else
        {
            yield return null;
        }

        float reward;
        float qdRawDelta = 0f, btRawDelta = 0f, plRawDelta = 0f;
        if (factoryEnv.useObservationWindow)
        {
            reward = factoryEnv.GetLastGlobalReward();
            plRawDelta = factoryEnv.GetPlRaw();
            qdRawDelta = factoryEnv.GetLastWindowQdDelta();
            btRawDelta = factoryEnv.GetLastWindowBtDelta();
        }
        else
        {
            reward = GetInstantReward();
            qdRawDelta = factoryEnv.GetCurrentQdDelta();
            btRawDelta = factoryEnv.GetCurrentBtDelta();
        }

        //float[] nextState = GetConcatenatedState();   // s_{t+1} after window
        //transitionStepCounter++;

        float[] nextState = GetConcatenatedGrid();
        float[] nextSpawnerRates = GetConcatenatedSpawnerRates();

        var msg = new TransitionMessage
        {
            action_id = actionId,
            node_id = nodeId,
            reward = reward,
            pl_raw_delta = plRawDelta,
            qd_raw_delta = qdRawDelta,
            bt_raw_delta = btRawDelta,
            state_t = state_t,
            state_tp1 = nextState,
            spawner_rates_t = lastSpawnerRates,
            spawner_rates_tp1 = nextSpawnerRates
        };
        string json = JsonUtility.ToJson(msg);
        if (debugLogs)
        {
            Debug.Log($"[DqnAgent] Transition {transitionStepCounter}: action={actionId}, node={nodeId}, reward={reward:F3}");
            Debug.Log($"[DqnAgent] state_t len={state_t.Length}, state_tp1 len={nextState.Length}");
        }

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

    // ------------------------------------------------------------
    // Action Request (now uses concatenated state)
    // ------------------------------------------------------------

    public IEnumerator CoRequestActionAndPickNode(List<int> candidates, float epsilon, Action<int, bool> onDone)
    {
        if (tcpClient == null)
        {
            int fallback = (candidates != null && candidates.Count > 0) ? candidates[UnityEngine.Random.Range(0, candidates.Count)] : -1;
            onDone?.Invoke(fallback, true);
            yield break;
        }

        if (candidates == null || candidates.Count == 0)
        {
            onDone?.Invoke(-1, true);
            yield break;
        }

        float[] stateVec = GetConcatenatedGrid();
        float[] spawnerVec = GetConcatenatedSpawnerRates();

        ActionRequestMessage req = new ActionRequestMessage
        {
            state = stateVec,
            spawner_rates = spawnerVec,
            candidate_node_ids = candidates.ToArray(),
            epsilon = epsilon
        };

        string json = JsonUtility.ToJson(req);
        if (debugLogs)
            Debug.Log($"[DqnAgent] action_request sent. candidates=[{string.Join(",", candidates)}]");

        waitingActionReply = true;
        hasLastActionReply = false;
        tcpClient.SendJsonLine(json);

        float timeout = 2.0f;
        float startTime = Time.time;
        while (waitingActionReply && (Time.time - startTime) < timeout)
            yield return null;

        if (!hasLastActionReply)
        {
            int fallback = candidates[UnityEngine.Random.Range(0, candidates.Count)];
            onDone?.Invoke(fallback, true);
            yield break;
        }

        onDone?.Invoke(lastChosenNodeId, lastIsRandom);
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

    // ------------------------------------------------------------
    // Reward helpers (unchanged)
    // ------------------------------------------------------------

    float GetInstantReward()
    {
        float PL, QD, BT, EC, RO, PLn, QDn, BTn, ECn, ROn;
        float r = factoryEnv.ComputeGlobalReward(out PL, out QD, out BT, out EC, out RO, out PLn, out QDn, out BTn, out ECn, out ROn);
        if (debugLogs)
            Debug.Log($"[DqnAgent] Instant Reward={r:F3}");
        return r;
    }

    // ------------------------------------------------------------
    // Debug Helpers
    // ------------------------------------------------------------

    void DebugLogState(string label, float[] state)
    {
        if (state == null) return;
        int len = state.Length;
        int n = Mathf.Min(len, maxStateElementsToLog);
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendFormat("[DqnAgent] {0} (len={1}) first {2}: [", label, len, n);
        for (int i = 0; i < n; i++)
        {
            sb.Append(state[i].ToString("0.000"));
            if (i < n - 1) sb.Append(", ");
        }
        if (len > n) sb.Append(" ...");
        sb.Append("]");
        Debug.Log(sb.ToString());
    }

    void DebugLogNodeSnapshot()
    {
        // optional – not modified
    }
}