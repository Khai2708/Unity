using System;
using System.Collections;
using System.Collections.Generic;
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
    private const int GRID_ROWS = 5;
    private const int GRID_COLS = 7;
    private const int NUM_CHANNELS = 8;

    [Header("History (N steps)")]
    public int historyLength = 3;
    public float snapshotInterval = 1.0f;

    [Header("Normalisation")]
    public float maxQueueCapacity = 50f;
    public float maxQueueCount = 50f;
    public float maxStateDuration = 100f;

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

    // Spawner rates
    private int numSpawners;
    private Queue<float[]> spawnerHistory = new Queue<float[]>();
    private float[] lastSpawnerRates;  // for s_t

    // New: 저장된 전체 보상
    private float lastTotalReward;

    void Start()
    {
        if (factoryEnv != null && factoryEnv.spawners != null)
            numSpawners = factoryEnv.spawners.Length;
        else
            numSpawners = 0;
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
        public float reward;                 // total reward (weighted sum)
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
        if (Time.time >= nextSnapshotTime)
        {
            TakeSnapshot();
            nextSnapshotTime = Time.time + snapshotInterval;
        }
    }

    // ------------------------------------------------------------
    // Snapshot & History Management
    // ------------------------------------------------------------

    void TakeSnapshot()
    {
        float[] currentGrid = BuildCurrentGridSnapshot();
        float[] currentRates = factoryEnv.GetCurrentSpawnerRates();

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

    float[] BuildCurrentGridSnapshot()
    {
        if (factoryEnv == null || factoryEnv.Nodes == null)
            return new float[GRID_ROWS * GRID_COLS * NUM_CHANNELS];

        var nodesDict = factoryEnv.Nodes;

        List<FactoryEnvManager.NodeData> tunnelNodes = new List<FactoryEnvManager.NodeData>();
        foreach (var kv in nodesDict)
        {
            if (!kv.Value.isSpawner && kv.Value.tunnel != null)
                tunnelNodes.Add(kv.Value);
        }

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

            float normCapacity = Mathf.Clamp01(node.queueCapacity / maxQueueCapacity);
            float normCount = Mathf.Clamp01(node.queueCount / maxQueueCount);
            float normDuration = Mathf.Clamp01(duration / maxStateDuration);

            cellCount[row, col]++;

            cellSum[row, col, 0] += normCapacity;
            cellSum[row, col, 1] += normCount;
            cellSum[row, col, 2] += (stateIdx == 0) ? 1 : 0;
            cellSum[row, col, 3] += (stateIdx == 1) ? 1 : 0;
            cellSum[row, col, 4] += (stateIdx == 2) ? 1 : 0;
            cellSum[row, col, 5] += (stateIdx == 3) ? 1 : 0;
            cellSum[row, col, 6] += normDuration;
        }

        float[] flat = new float[GRID_ROWS * GRID_COLS * NUM_CHANNELS];
        int idx = 0;
        for (int row = 0; row < GRID_ROWS; row++)
        {
            for (int col = 0; col < GRID_COLS; col++)
            {
                int cnt = cellCount[row, col];
                float invCnt = (cnt > 0) ? 1f / cnt : 0f;

                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                {
                    if (ch == 7)
                        flat[idx++] = (cnt > 0) ? 1f : 0f;
                    else
                        flat[idx++] = cellSum[row, col, ch] * invCnt;
                }
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

    float[] GetConcatenatedGrid()
    {
        int snapshotLength = GRID_ROWS * GRID_COLS * NUM_CHANNELS;
        int totalLength = historyLength * snapshotLength;
        float[] result = new float[totalLength];
        int offset = 0;
        foreach (var snap in snapshotHistory)
        {
            Array.Copy(snap, 0, result, offset, snapshotLength);
            offset += snapshotLength;
        }
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
    // Action Recording (with total reward)
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
            //if (logStackTraceOnDuplicateRecord)
                //Debug.Log(new StackTrace(true));
        }

        lastState = GetConcatenatedGrid();
        lastSpawnerRates = GetConcatenatedSpawnerRates();
        lastActionId = actionId;
        lastNodeId = nodeId;
        hasPendingTransition = true;

        // Compute total reward using the factory's weighted sum
        lastTotalReward = factoryEnv.ComputeTotalReward(nodeId);
        if (debugLogs)
            Debug.Log($"[DqnAgent] Total Reward for node {nodeId}: {lastTotalReward:F3}");
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

        StartCoroutine(CoWaitAndSend(actionId, nodeId, state_t));
    }

    IEnumerator CoWaitAndSend(int actionId, int nodeId, float[] state_t)
    {
        // 한 프레임 대기 (선택사항)
        yield return null;

        float reward = lastTotalReward;
        float plRawDelta = 0f;
        float qdRawDelta = 0f;
        float btRawDelta = 0f;

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
        transitionStepCounter++;
    }

    void DebugPrintSnapshot(float[] snapshot)
    {
        if (snapshot == null || snapshot.Length != GRID_ROWS * GRID_COLS * NUM_CHANNELS)
        {
            Debug.LogWarning("[DqnAgent] Cannot print snapshot: invalid length");
            return;
        }

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine($"[DqnAgent] Current Grid Snapshot (Rows={GRID_ROWS}, Cols={GRID_COLS}, Channels={NUM_CHANNELS}):");

        for (int row = 0; row < GRID_ROWS; row++)
        {
            sb.Append($"Row {row}: ");
            for (int col = 0; col < GRID_COLS; col++)
            {
                int baseIdx = (row * GRID_COLS + col) * NUM_CHANNELS;
                float exist = snapshot[baseIdx + 7];
                if (exist > 0.5f)
                {
                    float cap = snapshot[baseIdx];
                    float q = snapshot[baseIdx + 1];
                    float[] stateVals = new float[4];
                    for (int s = 0; s < 4; s++) stateVals[s] = snapshot[baseIdx + 2 + s];
                    int stateIdx = 0;
                    float maxVal = stateVals[0];
                    for (int s = 1; s < 4; s++)
                    {
                        if (stateVals[s] > maxVal)
                        {
                            maxVal = stateVals[s];
                            stateIdx = s;
                        }
                    }
                    string stateStr = stateIdx switch
                    {
                        0 => "RUN",
                        1 => "HALF",
                        2 => "HOLD",
                        3 => "FAULT",
                        _ => "?"
                    };
                    float dur = snapshot[baseIdx + 6];
                    sb.Append($"[{row},{col}: C={cap:F2} Q={q:F2} S={stateStr} D={dur:F2}] ");
                }
                else
                {
                    sb.Append($"[{row},{col}: empty] ");
                }
            }
            sb.AppendLine();
        }
        Debug.Log(sb.ToString());
    }

    // ------------------------------------------------------------
    // Action Request (unchanged)
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

        if (logStateVector && stateVec != null)
        {
            int snapshotLength = GRID_ROWS * GRID_COLS * NUM_CHANNELS;
            if (stateVec.Length == historyLength * snapshotLength)
            {
                float[] currentSnapshot = new float[snapshotLength];
                Array.Copy(stateVec, (historyLength - 1) * snapshotLength, currentSnapshot, 0, snapshotLength);
                DebugPrintSnapshot(currentSnapshot);
            }
            else
            {
                Debug.LogWarning($"[DqnAgent] stateVec length {stateVec.Length} does not match expected {historyLength * snapshotLength}");
            }
        }

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
}