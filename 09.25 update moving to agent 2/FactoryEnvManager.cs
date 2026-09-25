using System.Collections.Generic;
using System.Text;
using UnityEngine;

public class FactoryEnvManager : MonoBehaviour
{
    public void ResetScenarioAHoldTracking()
    {
    }

    public void NotifyRepairStart(int nodeId = -1)
    {
    }

    public void NotifyDecisionStepRewardFinalized(int nodeId = -1)
    {
    }

    [Header("simulationTimeScale")]
    [Tooltip("simulationTimeScale")]
    public float simulationTimeScale = 1f;

    [Header("secondsPerTU")]
    [Tooltip("secondsPerTU")]
    public float secondsPerTU = 0.5f;

    public float TUToSeconds(int tu)
    {
        return Mathf.Max(0, tu) * Mathf.Max(1e-6f, secondsPerTU);
    }

    public float GetSimulationTimeSeconds()
    {
        if (enableGlobalTUClock)
        {
            float sPerTU = Mathf.Max(1e-6f, secondsPerTU);
            return (_globalTU * sPerTU) + _tuAcc;
        }

        return Time.time;
    }

    public long GetCurrentTULong()
    {
        if (enableGlobalTUClock)
            return _globalTU;

        float sPerTU = Mathf.Max(1e-6f, secondsPerTU);
        return Mathf.FloorToInt(GetSimulationTimeSeconds() / sPerTU);
    }

    public int GetCurrentTU()
    {
        long tu = GetCurrentTULong();
        return (tu > int.MaxValue) ? int.MaxValue : (int)tu;
    }

    // ===== Global TU Clock =====
    [Header("Global TU Clock")]
    [Tooltip("Enable global TU clock accumulation in Update.")]
    public bool enableGlobalTUClock = true;

    [Tooltip("Use Time.unscaledDeltaTime for TU accumulation (ignores timeScale). ")]
    public bool tuUseUnscaledDeltaTime = false;

    // TU accumulator (starts from 0)
    private long _globalTU = 0;
    private float _tuAcc = 0f;

    /// <summary>Current global TU (starts from 0), independent from GetCurrentTU().</summary>
    public long GlobalTU => _globalTU;

    public int TotalExits => _globalExitCount;

    /// <summary>Elapsed seconds within the current TU.</summary>
    public float SecondsIntoCurrentTU => _tuAcc;

    private void TickTU()
    {
        if (!enableGlobalTUClock) return;

        float dt = tuUseUnscaledDeltaTime ? Time.unscaledDeltaTime : Time.deltaTime;
        float sPerTU = Mathf.Max(1e-6f, secondsPerTU);

        _tuAcc += Mathf.Max(0f, dt);

        while (_tuAcc >= sPerTU)
        {
            _tuAcc -= sPerTU;
            _globalTU++;
        }
    }

    public static FactoryEnvManager Instance { get; private set; }

    [Header("spawners")]
    public ProductSpawner[] spawners;
    public TunnelController[] tunnels;

    private Dictionary<TunnelController, int> _prevSinkExitCountsInstant
        = new Dictionary<TunnelController, int>();

    private Dictionary<int, NodeData> nodes = new Dictionary<int, NodeData>();

    private Dictionary<int, List<int>> adjacency = new Dictionary<int, List<int>>();
    private Dictionary<int, List<int>> reverseAdjacency = new Dictionary<int, List<int>>();
    private Dictionary<(int from, int to), float> edgeProbabilities = new Dictionary<(int, int), float>();
    private Dictionary<int, float> _initialSpawnerRates = new Dictionary<int, float>();
    private Dictionary<int, float> _decisionSpawnerRates = new Dictionary<int, float>();

    private Dictionary<TunnelController, TunnelController.TunnelState> _lastTunnelStates
        = new Dictionary<TunnelController, TunnelController.TunnelState>();

    [Header("minFaultReward")]
    [Tooltip("minFaultReward")]
    public float minFaultReward = 1f;

    [Tooltip("maxFaultReward")]
    public float maxFaultReward = 5f;

    private Dictionary<TunnelController, float> faultRewards
        = new Dictionary<TunnelController, float>();

    [Header("debugLogGlobalReward")]
    [Tooltip("debugLogGlobalReward")]
    public bool debugLogGlobalReward = true;

    [Header("Reward Weights (Active)")]
    [Tooltip("Weight for Assembly Reward.")]
    public float W1_AR = 0.7f;

    [Tooltip("Weight for Queue Reward.")]
    public float W2_QR = 0.2f;

    [Tooltip("Weight for Energy Cost.")]
    public float W3_EC = 0.1f;

    private float _lastGlobalReward = 0f;
    private float _lastAssemblyReward = 0f;
    private float _lastEnergyReward = 0f;

    //[Header("Wrong Selection Penalty")]
   // [Tooltip("정상 노드를 잘못 선택했을 때 부과되는 패널티 로그를 출력할지 여부")]
    //public bool debugWrongSelectionPenalty = true;
    //private float _lastWrongSelectionPenalty = 0f;

    private int _globalExitCount = 0;

    private class DecisionSnapshot
    {
        public int decisionNodeId;
        public int decisionTU;

        public readonly HashSet<int> queueRewardNodeSet = new HashSet<int>();
        public float queueSumAtDecision;
        public float queueCapacitySum;
        public bool hasQueueRewardSnapshot;

        public readonly HashSet<int> faultSnapshotAtDecision = new HashSet<int>();

        public int energyDecisionTU;
        public int energyArrivalTU = -1;
        public int energyMoveTU;
        public int energyRepairTU;
        public bool hasEnergyObservation;
        public bool hasEnergyArrivalSnapshot;
    }

    private readonly Dictionary<int, DecisionSnapshot> _pendingDecisions = new Dictionary<int, DecisionSnapshot>();

    //[Header("Hit Selection Bonus")]
    //[Tooltip("고장 노드를 정확히 선택했을 때 주는 보너스 로그를 출력할지 여부")]
    //public bool debugHitSelectionBonus = true;
    // private float _lastHitSelectionBonus = 0f;

    [Header("debugQueueReward")]
    [Tooltip("debugQueueReward")]
    public bool debugQueueReward = true;

    [Header("debugAssemblyReward")]
    [Tooltip("debugAssemblyReward")]
    public bool debugAssemblyReward = true;

    [Header("debugEnergyReward")]
    [Tooltip("debugEnergyReward")]
    public bool debugEnergyReward = true;

    [Header("Energy Reward")]
    [Tooltip("Move energy consumed per TU.")]
    public float moveEnergyPerTU = 1f;

    [Tooltip("Repair energy consumed per TU.")]
    public float repairEnergyPerTU = 1f;

    [Tooltip("Maximum move TU used for EC normalization.")]
    public int maxMoveTU = 30;

    [Tooltip("Maximum repair TU used for EC normalization.")]
    public int maxRepairTU = 15;


    public void BeginQueueRewardObservation(int decisionNodeId)
    {
        var snap = new DecisionSnapshot
        {
            decisionNodeId = decisionNodeId,
            decisionTU = GetCurrentTU(),
            energyDecisionTU = GetCurrentTU(),
            hasEnergyObservation = (decisionNodeId >= 0),
        };

        if (debugEnergyReward && snap.hasEnergyObservation)
        {
            Debug.Log($"[EnergyReward][Decision] node={decisionNodeId} decision_tu={snap.energyDecisionTU} moveUnit={moveEnergyPerTU:F3} repairUnit={repairEnergyPerTU:F3} maxMoveTU={maxMoveTU} maxRepairTU={maxRepairTU}");
        }

        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;
                if (t.nodeId < 0) continue;
                if (t.State == TunnelController.TunnelState.FAULT)
                    snap.faultSnapshotAtDecision.Add(t.nodeId);
            }
        }

        if (decisionNodeId >= 0 && nodes.ContainsKey(decisionNodeId))
        {
            Queue<int> q = new Queue<int>();
            HashSet<int> visited = new HashSet<int>();
            visited.Add(decisionNodeId);
            q.Enqueue(decisionNodeId);

            while (q.Count > 0)
            {
                int cur = q.Dequeue();

                if (nodes.TryGetValue(cur, out var nd) && !nd.isSpawner && nd.tunnel != null)
                    snap.queueRewardNodeSet.Add(cur);

                if (!reverseAdjacency.TryGetValue(cur, out var prevs) || prevs == null) continue;

                foreach (var prev in prevs)
                {
                    if (!nodes.ContainsKey(prev)) continue;
                    if (visited.Add(prev)) q.Enqueue(prev);
                }
            }

            if (snap.queueRewardNodeSet.Count > 0)
            {
                float qAtT = 0f;
                float capSum = 0f;
                foreach (var nid in snap.queueRewardNodeSet)
                {
                    if (!nodes.TryGetValue(nid, out var nd)) continue;
                    int qNow = nd.queueCount;
                    int capNow = nd.queueCapacity;

                    if (nd.tunnel != null && nd.tunnel.queue != null)
                    {
                        qNow = nd.tunnel.queue.Count;
                        capNow = nd.tunnel.queue.Capacity;
                        nd.queueCount = qNow;
                        nd.queueCapacity = capNow;
                    }

                    qAtT += qNow;
                    capSum += Mathf.Max(0f, capNow);
                }

                snap.queueSumAtDecision = qAtT;
                snap.queueCapacitySum = capSum;
                snap.hasQueueRewardSnapshot = (snap.queueCapacitySum > 0f);

                if (debugQueueReward)
                {
                    var ids = new List<int>(snap.queueRewardNodeSet);
                    ids.Sort();
                    string idText = string.Join(",", ids);
                    Debug.Log(
                        $"[QueueReward][Decision] node={decisionNodeId} setCount={snap.queueRewardNodeSet.Count} " +
                        $"tu={snap.decisionTU} q_t={snap.queueSumAtDecision:F1} capSum={snap.queueCapacitySum:F1} hasSnapshot={snap.hasQueueRewardSnapshot} " +
                        $"nodes=[{idText}]"
                    );

                    if (snap.queueRewardNodeSet.Count <= 1)
                    {
                        Debug.LogWarning($"[QueueReward][Begin] upstream set is small (count={snap.queueRewardNodeSet.Count}). Check graph wiring from spawners to node={decisionNodeId}.");
                    }
                }
            }
        }

        _pendingDecisions[decisionNodeId] = snap; // per-decision entry -- no cross-robot overwrite
    }

    // ---- METHOD: ConsumeQueueRewardNorm (now takes nodeId) ----
    private float ConsumeQueueRewardNorm(int nodeId, out float deltaQueueRaw, out float capacitySum)
    {
        deltaQueueRaw = 0f;
        capacitySum = 0f;

        if (!_pendingDecisions.TryGetValue(nodeId, out var snap))
        {
            if (debugQueueReward)
                Debug.LogWarning($"[QueueReward][Consume] skipped: no pending decision for node={nodeId}");
            return 0f;
        }

        capacitySum = snap.queueCapacitySum;

        if (!snap.hasQueueRewardSnapshot || snap.queueRewardNodeSet.Count == 0 || snap.queueCapacitySum <= 0f)
        {
            if (debugQueueReward)
            {
                Debug.LogWarning($"[QueueReward][Consume] skipped snapshot={snap.hasQueueRewardSnapshot} setCount={snap.queueRewardNodeSet.Count} capSum={snap.queueCapacitySum:F1}");
            }
            return 0f;
        }

        float qAtTp1 = 0f;
        foreach (var nid in snap.queueRewardNodeSet)
        {
            if (!nodes.TryGetValue(nid, out var nd)) continue;
            int qNow = nd.queueCount;
            if (nd.tunnel != null && nd.tunnel.queue != null)
            {
                qNow = nd.tunnel.queue.Count;
                nd.queueCount = qNow;
            }
            qAtTp1 += qNow;
        }

        float rawDelta = qAtTp1 - snap.queueSumAtDecision;
        if (rawDelta < 0f && debugQueueReward)
        {
            Debug.LogWarning($"[QueueReward][Consume] negative dQ detected node={nodeId} rawDelta={rawDelta:F1} -> clamped to 0");
        }

        deltaQueueRaw = Mathf.Max(0f, rawDelta);
        float norm = deltaQueueRaw / snap.queueCapacitySum;

        if (debugQueueReward)
        {
            int tuAfter = GetCurrentTU();
            Debug.Log(
                $"[QueueReward][AfterRepair] node={nodeId} decision_tu={snap.decisionTU} after_tu={tuAfter} " +
                $"q_t={snap.queueSumAtDecision:F1} q_tp1={qAtTp1:F1} dQ={deltaQueueRaw:F1} capSum={snap.queueCapacitySum:F1} ratio={norm:F4}"
            );
        }

        return Mathf.Clamp(norm, -1f, 1f);
    }

    // ---- METHOD: NotifyRobotArrivedAtDecisionTarget (now looks up by nodeId) ----
    public void NotifyRobotArrivedAtDecisionTarget(int nodeId, int repairTU)
    {
        if (!_pendingDecisions.TryGetValue(nodeId, out var snap) || !snap.hasEnergyObservation)
        {
            if (debugEnergyReward)
                Debug.LogWarning($"[EnergyReward][Arrival] skipped node={nodeId} because no decision snapshot exists.");
            return;
        }

        snap.energyArrivalTU = GetCurrentTU();
        snap.energyRepairTU = Mathf.Max(0, repairTU);
        snap.energyMoveTU = Mathf.Max(0, snap.energyArrivalTU - snap.energyDecisionTU);
        snap.hasEnergyArrivalSnapshot = true;

        if (debugEnergyReward)
        {
            Debug.Log(
                $"[EnergyReward][Arrival] node={nodeId} decision_tu={snap.energyDecisionTU} arrival_tu={snap.energyArrivalTU} " +
                $"move_tu={snap.energyMoveTU} repair_tu={snap.energyRepairTU}"
            );
        }
    }

    // ---- METHOD: ConsumeEnergyRewardNorm (now takes nodeId) ----
    private float ConsumeEnergyRewardNorm(int nodeId, out float moveTuRaw, out float repairTuRaw, out float moveEnergy, out float repairEnergy, out float energyRaw)
    {
        moveTuRaw = 0f; repairTuRaw = 0f; moveEnergy = 0f; repairEnergy = 0f; energyRaw = 0f;

        if (!_pendingDecisions.TryGetValue(nodeId, out var snap) || !snap.hasEnergyObservation)
        {
            if (debugEnergyReward)
                Debug.LogWarning($"[EnergyReward][Consume] skipped: no decision observation for node={nodeId}.");
            return 0f;
        }

        if (!snap.hasEnergyArrivalSnapshot && debugEnergyReward)
        {
            Debug.LogWarning($"[EnergyReward][Consume] missing arrival snapshot for node={nodeId}. Using move_tu=0.");
        }

        moveTuRaw = Mathf.Max(0f, snap.energyMoveTU);
        repairTuRaw = Mathf.Max(0f, snap.energyRepairTU);
        moveEnergy = moveTuRaw * Mathf.Max(0f, moveEnergyPerTU);
        repairEnergy = repairTuRaw * Mathf.Max(0f, repairEnergyPerTU);
        energyRaw = moveEnergy + repairEnergy;

        float maxEnergy =
            (Mathf.Max(0, maxMoveTU) * Mathf.Max(0f, moveEnergyPerTU)) +
            (Mathf.Max(0, maxRepairTU) * Mathf.Max(0f, repairEnergyPerTU));
        float norm = (maxEnergy > 0f) ? (energyRaw / maxEnergy) : 0f;

        if (debugEnergyReward)
        {
            Debug.Log(
                $"[EnergyReward][Reward] node={nodeId} decision_tu={snap.energyDecisionTU} arrival_tu={snap.energyArrivalTU} " +
                $"move_tu={moveTuRaw:F1} repair_tu={repairTuRaw:F1} Emove={moveEnergy:F4} Erepair={repairEnergy:F4} raw={energyRaw:F4} norm={norm:F4}"
            );
        }

        return Mathf.Clamp01(norm);
    }

    public bool TryPeekEnergyBreakdown(int nodeId, out int moveTU, out int repairTU)
    {
        moveTU = 0;
        repairTU = 0;

        if (!_pendingDecisions.TryGetValue(nodeId, out var snap) || !snap.hasEnergyObservation)
            return false;

        if (!snap.hasEnergyArrivalSnapshot)
            return false;

        moveTU = Mathf.Max(0, snap.energyMoveTU);
        repairTU = Mathf.Max(0, snap.energyRepairTU);
        return true;
    }

    public void RegisterProductExit()
    {
        _globalExitCount++;
    }

    private int _qdAtObsStart = 0;
    private int _btAtObsStart = 0;

    float _sumQD, _sumFT, _sumBT, _sumEC, _sumRO;
    int _sampleCount;

    Dictionary<TunnelController, int> _sinkStartCounts = new Dictionary<TunnelController, int>();

    [System.Serializable]
    public class NodeData
    {
        public int nodeId;
        public string name;

        public bool isSpawner;
        public ProductSpawner spawner;
        public TunnelController tunnel;

        public TunnelController.TunnelState tunnelState;
        public int queueCount;
        public int queueCapacity;
    }

    public IReadOnlyDictionary<int, NodeData> Nodes => nodes;
    public IReadOnlyDictionary<int, List<int>> Adjacency => adjacency;

    [Header("debugLogOnBuild")]
    [Tooltip("debugLogOnBuild")]
    public bool debugLogOnBuild = true;

    [Header("debugCompactState")]
    [Tooltip("debugCompactState")]
    public bool debugCompactState = false;

    [Tooltip("debugCompactInterval")]
    public float debugCompactInterval = 1f;

    private float _nextCompactLogTime = 0f;

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[Log] warning");
            Destroy(this);
            return;
        }

        Instance = this;

        if (spawners == null || spawners.Length == 0)
            spawners = FindObjectsOfType<ProductSpawner>();

        if (tunnels == null || tunnels.Length == 0)
            tunnels = FindObjectsOfType<TunnelController>();

        BuildNodeIndex();
        BuildGraphEdges();
        InitializeSpawnerBaselineRates();

        if (debugLogOnBuild)
        {
            DumpGraphToLog();
        }

        _globalTU = 0;
        _tuAcc = 0f;
        _nextCompactLogTime = GetSimulationTimeSeconds() + debugCompactInterval;
        ApplyTimeScale();
    }

    void ApplyTimeScale()
    {
        Time.timeScale = simulationTimeScale;
        Time.fixedDeltaTime = 0.02f * Time.timeScale;
    }

    void Update()
    {
        TickTU();
        UpdateNodeStates();

        if (debugCompactState && GetSimulationTimeSeconds() >= _nextCompactLogTime)
        {
            DumpCompactStatesToLog();
            _nextCompactLogTime = GetSimulationTimeSeconds() + Mathf.Max(0.1f, debugCompactInterval);
        }
    }

    public void SyncNodeStatesImmediate()
    {
        UpdateNodeStates();
    }

    public void BeginRewardObservation()
    {
    }

    void BuildNodeIndex()
    {
        nodes.Clear();
        _lastTunnelStates.Clear();
        faultRewards.Clear();

        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;

                int id = sp.nodeId;
                if (id < 0)
                {
                    Debug.LogWarning("[Log] warning");
                    continue;
                }

                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning("[Log] warning");
                    continue;
                }

                NodeData data = new NodeData
                {
                    nodeId = id,
                    name = sp.name,
                    isSpawner = true,
                    spawner = sp,
                    tunnel = null,
                    tunnelState = TunnelController.TunnelState.RUN,
                    queueCount = 0,
                    queueCapacity = 0
                };

                nodes.Add(id, data);
            }
        }

        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                int id = t.nodeId;
                if (id < 0)
                {
                    Debug.LogWarning("[Log] warning");
                    continue;
                }

                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning("[Log] warning");
                    continue;
                }

                int qCount = 0;
                int qCap = 0;

                if (t.queue != null)
                {
                    qCount = t.queue.Count;
                    qCap = t.queue.Capacity;
                }

                NodeData data = new NodeData
                {
                    nodeId = id,
                    name = t.name,
                    isSpawner = false,
                    spawner = null,
                    tunnel = t,
                    tunnelState = t.State,
                    queueCount = qCount,
                    queueCapacity = qCap
                };

                nodes.Add(id, data);
                _lastTunnelStates[t] = t.State;
            }
        }
    }

    void BuildGraphEdges()
    {
        adjacency.Clear();
        reverseAdjacency.Clear();
        edgeProbabilities.Clear();

        Dictionary<(int from, int to), float> junctionEdgeProb = new Dictionary<(int, int), float>();
        var junctions = FindObjectsOfType<JunctionPoint>();
        if (junctions != null)
        {
            foreach (var jp in junctions)
            {
                if (jp == null || jp.parentTunnel == null || jp.branches == null || jp.branches.Length == 0) continue;
                int fromId = jp.parentTunnel.nodeId;
                if (fromId < 0) continue;

                List<(int toId, float w)> valid = new List<(int toId, float w)>();
                float posSum = 0f;
                foreach (var b in jp.branches)
                {
                    if (b == null || b.downstreamTunnel == null) continue;
                    int toId = b.downstreamTunnel.nodeId;
                    if (toId < 0) continue;
                    float w = Mathf.Max(0f, b.baseProbability);
                    valid.Add((toId, w));
                    posSum += w;
                }

                if (valid.Count == 0) continue;
                if (posSum <= 0f)
                {
                    float u = 1f / valid.Count;
                    foreach (var v in valid) junctionEdgeProb[(fromId, v.toId)] = u;
                }
                else
                {
                    foreach (var v in valid) junctionEdgeProb[(fromId, v.toId)] = v.w / posSum;
                }
            }
        }

        System.Action<int, int> addDirectedEdge = (fromId, toId) =>
        {
            if (!adjacency.TryGetValue(fromId, out var nextList))
            {
                nextList = new List<int>();
                adjacency.Add(fromId, nextList);
            }
            if (!nextList.Contains(toId))
                nextList.Add(toId);

            if (!reverseAdjacency.TryGetValue(toId, out var prevList))
            {
                prevList = new List<int>();
                reverseAdjacency.Add(toId, prevList);
            }
            if (!prevList.Contains(fromId))
                prevList.Add(fromId);
        };

        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;
                int fromId = sp.nodeId;
                if (fromId < 0) continue;
                if (!nodes.ContainsKey(fromId)) continue;

                if (sp.firstTunnels != null)
                {
                    int validCount = 0;
                    float specifiedPositiveSum = 0f;
                    foreach (var t in sp.firstTunnels)
                    {
                        if (t == null) continue;
                        int toId = t.nodeId;
                        if (toId < 0 || !nodes.ContainsKey(toId)) continue;
                        validCount++;
                    }
                    if (sp.branchProbabilities != null)
                    {
                        int n = Mathf.Min(sp.branchProbabilities.Length, sp.firstTunnels.Length);
                        for (int i = 0; i < n; i++)
                        {
                            specifiedPositiveSum += Mathf.Max(0f, sp.branchProbabilities[i]);
                        }
                    }
                    float uniformP = (validCount > 0) ? (1f / validCount) : 0f;

                    for (int i = 0; i < sp.firstTunnels.Length; i++)
                    {
                        var t = sp.firstTunnels[i];
                        if (t == null) continue;
                        int toId = t.nodeId;
                        if (toId < 0) continue;
                        if (!nodes.ContainsKey(toId)) continue;

                        addDirectedEdge(fromId, toId);
                        float p = uniformP;
                        if (sp.branchProbabilities != null && i < sp.branchProbabilities.Length && specifiedPositiveSum > 0f)
                        {
                            p = Mathf.Max(0f, sp.branchProbabilities[i]) / specifiedPositiveSum;
                        }
                        edgeProbabilities[(fromId, toId)] = p;
                    }
                }
            }
        }

        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;
                int fromId = t.nodeId;
                if (fromId < 0) continue;
                if (!nodes.ContainsKey(fromId)) continue;

                var next = t.nextTunnelsForGraph;
                if (next == null) continue;

                int validCount = 0;
                float specifiedPositiveSum = 0f;
                foreach (var child in next)
                {
                    if (child == null) continue;
                    int toId = child.nodeId;
                    if (toId < 0 || !nodes.ContainsKey(toId)) continue;
                    validCount++;
                }
                if (t.branchProbabilities != null)
                {
                    int n = Mathf.Min(t.branchProbabilities.Length, next.Length);
                    for (int i = 0; i < n; i++)
                    {
                        specifiedPositiveSum += Mathf.Max(0f, t.branchProbabilities[i]);
                    }
                }
                float uniformP = (validCount > 0) ? (1f / validCount) : 0f;

                for (int i = 0; i < next.Length; i++)
                {
                    var child = next[i];
                    if (child == null) continue;
                    int toId = child.nodeId;
                    if (toId < 0) continue;
                    if (!nodes.ContainsKey(toId)) continue;

                    addDirectedEdge(fromId, toId);
                    float p;
                    if (junctionEdgeProb.TryGetValue((fromId, toId), out var jpP))
                    {
                        p = jpP;
                    }
                    else
                    {
                        p = uniformP;
                        if (t.branchProbabilities != null && i < t.branchProbabilities.Length && specifiedPositiveSum > 0f)
                        {
                            p = Mathf.Max(0f, t.branchProbabilities[i]) / specifiedPositiveSum;
                        }
                    }
                    edgeProbabilities[(fromId, toId)] = p;
                }
            }
        }
    }

    private void InitializeSpawnerBaselineRates()
    {
        _initialSpawnerRates.Clear();
        if (spawners == null) return;

        foreach (var sp in spawners)
        {
            if (sp == null) continue;
            float baseRate = sp.GetNominalSpawnRatePerSec();
            _initialSpawnerRates[sp.nodeId] = baseRate;
        }
    }

    private float GetCurrentSpawnerRate(ProductSpawner sp)
    {
        if (sp == null) return 0f;
        return sp.GetEffectiveSpawnRatePerSec();
    }

    private bool IsPassableForFlow(int nodeId)
    {
        if (!nodes.TryGetValue(nodeId, out var nd)) return false;
        if (nd.isSpawner) return true;
        if (nd.tunnel == null) return false;
        var st = nd.tunnel.State;
        if (st == TunnelController.TunnelState.FAULT) return false;
        if (st == TunnelController.TunnelState.HOLD) return false;
        return true;
    }

    private bool HasAnyPassablePath(int nodeId, HashSet<int> visited)
    {
        if (!IsPassableForFlow(nodeId)) return false;
        if (!visited.Add(nodeId)) return false;

        if (!adjacency.TryGetValue(nodeId, out var nexts) || nexts == null || nexts.Count == 0)
        {
            visited.Remove(nodeId);
            return true;
        }

        foreach (var nxt in nexts)
        {
            if (HasAnyPassablePath(nxt, visited))
            {
                visited.Remove(nodeId);
                return true;
            }
        }

        visited.Remove(nodeId);
        return false;
    }

    private float ComputeEffectiveSpawnerRateAtDecision(ProductSpawner sp)
    {
        if (sp == null) return 0f;
        return sp.GetRewardSpawnRatePerSec();
    }

    private void CaptureDecisionSpawnerRates()
    {
        if (spawners == null) return;
        foreach (var sp in spawners)
        {
            if (sp == null) continue;
            float skt = ComputeEffectiveSpawnerRateAtDecision(sp);
            _decisionSpawnerRates[sp.nodeId] = skt;
            if (debugLogGlobalReward)
            {
                float sk0 = _initialSpawnerRates.TryGetValue(sp.nodeId, out var b) ? b : 0f;
                Debug.Log($"[AssemblyReward][SkCapture] src={sp.nodeId} capture_tu={GetCurrentTU()} Sk0={sk0:F4} Skt={skt:F4} hold={sp.IsHold} half={sp.IsHalfHold}");
            }
        }
    }

    public float[] GetCurrentSpawnerRates()
    {
        if (spawners == null || spawners.Length == 0)
            return new float[0];

        float[] rates = new float[spawners.Length];

        for (int i = 0; i < spawners.Length; i++)
        {
            var sp = spawners[i];
            if (sp == null)
            {
                rates[i] = 0f;
                continue;
            }

            rates[i] = GetCurrentSpawnerRate(sp);
        }
        return rates;
    }

    private HashSet<int> GetUpstreamSpawnerIds(int nodeId)
    {
        HashSet<int> result = new HashSet<int>();
        if (!nodes.ContainsKey(nodeId)) return result;

        Queue<int> q = new Queue<int>();
        HashSet<int> visited = new HashSet<int>();
        q.Enqueue(nodeId);
        visited.Add(nodeId);

        while (q.Count > 0)
        {
            int cur = q.Dequeue();
            if (!reverseAdjacency.TryGetValue(cur, out var prevs) || prevs == null) continue;

            foreach (var prev in prevs)
            {
                if (!nodes.TryGetValue(prev, out var nd)) continue;
                if (nd.isSpawner) result.Add(prev);
                if (visited.Add(prev)) q.Enqueue(prev);
            }
        }

        return result;
    }

    private bool TryGetPathProbabilityProduct(int sourceId, int targetId, out float product, out string edgeProbTrace)
    {
        product = 0f;
        edgeProbTrace = "";
        List<float> probs = new List<float>();
        bool found = DfsFindSinglePathProduct(sourceId, targetId, new HashSet<int>(), 1f, probs, out product);
        if (!found) return false;

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < probs.Count; i++)
        {
            if (i > 0) sb.Append(" * ");
            sb.Append(probs[i].ToString("F3"));
        }
        edgeProbTrace = sb.ToString();
        return true;
    }

    private bool DfsFindSinglePathProduct(
        int cur,
        int target,
        HashSet<int> visited,
        float probSoFar,
        List<float> probTrace,
        out float outProb)
    {
        if (cur == target)
        {
            outProb = probSoFar;
            return true;
        }

        if (!visited.Add(cur))
        {
            outProb = 0f;
            return false;
        }

        if (!adjacency.TryGetValue(cur, out var nexts) || nexts == null)
        {
            visited.Remove(cur);
            outProb = 0f;
            return false;
        }

        foreach (var nxt in nexts)
        {
            if (!nodes.ContainsKey(nxt)) continue;
            float edgeP = edgeProbabilities.TryGetValue((cur, nxt), out var p) ? p : 0f;
            if (edgeP <= 0f) continue;

            probTrace.Add(edgeP);
            if (DfsFindSinglePathProduct(nxt, target, visited, probSoFar * edgeP, probTrace, out outProb))
            {
                visited.Remove(cur);
                return true;
            }
            probTrace.RemoveAt(probTrace.Count - 1);
        }

        visited.Remove(cur);
        outProb = 0f;
        return false;
    }

    // ---- METHODS: fault-lookup helpers now take an explicit faultSnapshot ----
    private bool IsFaultTunnelNodeAtDecision(int nodeId, HashSet<int> faultSnapshot)
    {
        return faultSnapshot.Contains(nodeId);
    }

    // User definition (inverted I):
    // I=0 when decision node is closest fault to source.
    private bool IsClosestFaultToSource(int sourceId, int decisionNodeId, HashSet<int> faultSnapshot)
    {
        return ExistsPathWithoutOtherFault(sourceId, decisionNodeId, new HashSet<int>(), false, faultSnapshot);
    }

    private bool ExistsPathWithoutOtherFault(int cur, int target, HashSet<int> visited, bool hasOtherFault, HashSet<int> faultSnapshot)
    {
        if (cur == target) return !hasOtherFault;
        if (!visited.Add(cur)) return false;

        if (!adjacency.TryGetValue(cur, out var nexts) || nexts == null)
        {
            visited.Remove(cur);
            return false;
        }

        foreach (var nxt in nexts)
        {
            if (!nodes.ContainsKey(nxt)) continue;

            bool nextHasFault = hasOtherFault;
            if (nxt != target && IsFaultTunnelNodeAtDecision(nxt, faultSnapshot))
                nextHasFault = true;

            if (ExistsPathWithoutOtherFault(nxt, target, visited, nextHasFault, faultSnapshot))
            {
                visited.Remove(cur);
                return true;
            }
        }

        visited.Remove(cur);
        return false;
    }

    // ---- METHOD: ComputeAssemblyRewardForDecisionNode (reads snapshot's fault set) ----
    private float ComputeAssemblyRewardForDecisionNode(int decisionNodeId)
    {
        CaptureDecisionSpawnerRates();

        if (decisionNodeId < 0 || !nodes.ContainsKey(decisionNodeId)) return 0f;

        HashSet<int> faultSnapshot = _pendingDecisions.TryGetValue(decisionNodeId, out var snap)
            ? snap.faultSnapshotAtDecision
            : new HashSet<int>();

        var sourceIds = GetUpstreamSpawnerIds(decisionNodeId);
        float total = 0f;

        foreach (var sourceId in sourceIds)
        {
            if (!TryGetPathProbabilityProduct(sourceId, decisionNodeId, out float pathProb, out string probTrace))
                continue;

            float sk0 = _initialSpawnerRates.TryGetValue(sourceId, out var b) ? b : 0f;
            float skt;
            if (!_decisionSpawnerRates.TryGetValue(sourceId, out skt))
            {
                if (nodes.TryGetValue(sourceId, out var srcNd) && srcNd.isSpawner && srcNd.spawner != null)
                    skt = ComputeEffectiveSpawnerRateAtDecision(srcNd.spawner);
                else
                    skt = 0f;
            }

            bool closest = IsClosestFaultToSource(sourceId, decisionNodeId, faultSnapshot);
            int I = closest ? 0 : 1;
            bool useSk0Fallback = Mathf.Approximately(sk0, skt);
            float termValue = (I == 0) ? (useSk0Fallback ? sk0 : (sk0 - skt)) : skt;
            float contrib = termValue * pathProb;
            total += contrib;

            if (debugLogGlobalReward)
            {
                Debug.Log(
                    $"[AssemblyReward][Src] decision={decisionNodeId} src={sourceId} I={I} closest={closest} " +
                    $"Sk0={sk0:F4} Skt={skt:F4} pathP={pathProb:F4} (edges={probTrace}) " +
                    $"term={(I == 0 ? (useSk0Fallback ? "Sk0(eq)" : "(Sk0-Skt)") : "Skt")}={termValue:F4} contrib={contrib:F4}"
                );
            }
        }

        if (debugLogGlobalReward)
            Debug.Log($"[AssemblyReward][Total] decision={decisionNodeId} AR={total:F4}");

        return total;
    }

    void UpdateNodeStates()
    {
        if (tunnels == null) return;

        foreach (var t in tunnels)
        {
            if (t == null) continue;
            int id = t.nodeId;
            if (id < 0) continue;
            if (!nodes.TryGetValue(id, out var data)) continue;

            if (!data.isSpawner)
            {
                var currentState = t.State;

                if (_lastTunnelStates.TryGetValue(t, out var prevState))
                {
                    if (prevState != TunnelController.TunnelState.FAULT &&
                        currentState == TunnelController.TunnelState.FAULT)
                    {
                        OnTunnelFailed(t);
                    }
                    else if (prevState == TunnelController.TunnelState.FAULT &&
                             currentState != TunnelController.TunnelState.FAULT)
                    {
                        OnTunnelRepaired(t);
                    }
                }

                _lastTunnelStates[t] = currentState;

                data.tunnelState = currentState;

                if (t.queue != null)
                {
                    data.queueCount = t.queue.Count;
                    data.queueCapacity = t.queue.Capacity;
                }
                else
                {
                    data.queueCount = 0;
                    data.queueCapacity = 0;
                }
            }
        }
    }

    void OnTunnelFailed(TunnelController t)
    {
        float reward = Random.Range(minFaultReward, maxFaultReward);
        faultRewards[t] = reward;
    }

    void OnTunnelRepaired(TunnelController t)
    {
        if (faultRewards.ContainsKey(t))
        {
            faultRewards.Remove(t);
        }
    }

    public TunnelController GetBestFaultyTunnel()
    {
        TunnelController best = null;
        float bestReward = float.NegativeInfinity;

        foreach (var kvp in faultRewards)
        {
            if (kvp.Value > bestReward)
            {
                bestReward = kvp.Value;
                best = kvp.Key;
            }
        }

        return best;
    }

    // ---- METHOD: ComputeGlobalReward -- now takes nodeId, no hit/wrong terms ----
    public float ComputeGlobalReward(int nodeId, out float QD, out float QD_norm)
    {
        int decisionNodeId = nodeId;

        float queueCapSum;
        float queueDeltaRaw;
        float queueRawNorm = ConsumeQueueRewardNorm(decisionNodeId, out queueDeltaRaw, out queueCapSum);
        QD = queueRawNorm;
        QD_norm = Mathf.Clamp01(queueRawNorm);

        float moveTuRaw, repairTuRaw, moveEnergyRaw, repairEnergyRaw, ecRaw;
        float ecNorm = ConsumeEnergyRewardNorm(decisionNodeId, out moveTuRaw, out repairTuRaw, out moveEnergyRaw, out repairEnergyRaw, out ecRaw);
        _lastEnergyReward = ecNorm;

        float assemblyReward = ComputeAssemblyRewardForDecisionNode(decisionNodeId);
        _lastAssemblyReward = assemblyReward;

        float reward = (W2_QR * QD_norm) + (W1_AR * assemblyReward) - (W3_EC * ecNorm);
        _lastGlobalReward = reward;

        if (debugLogGlobalReward)
        {
            Debug.Log(
                $"[QueueReward][Reward] node={decisionNodeId} deltaQueue={queueDeltaRaw:F1} capSum={queueCapSum:F1} " +
                $"raw={queueRawNorm:F4} norm={QD_norm:F4} wQR={W2_QR:F3} wAR={W1_AR:F3} wEC={W3_EC:F3} " +
                $"assembly={assemblyReward:F4} ec={ecNorm:F4} reward={reward:F4}"
            );

            Debug.Log(
                $"[TotalReward] node={decisionNodeId} total={reward:F4} queue={QD_norm:F4} assembly={assemblyReward:F4} ec={ecNorm:F4} " +
                $"move_tu={moveTuRaw:F1} repair_tu={repairTuRaw:F1}"
            );
        }

        _pendingDecisions.Remove(decisionNodeId); // fully consumed -- no leak, no stale reuse
        return reward;
    }

    public float GetLastGlobalReward() => _lastGlobalReward;
    public float GetLastAssemblyReward() => _lastAssemblyReward;
    public float GetLastEnergyReward() => _lastEnergyReward;
    

    void DumpGraphToLog()
    {
        foreach (var pair in adjacency)
        {
            int from = pair.Key;
            string fromName = nodes.TryGetValue(from, out var n) ? n.name : "Unknown";

            var list = pair.Value;
            string targets = "";

            for (int i = 0; i < list.Count; i++)
            {
                int to = list[i];
                string toName = nodes.TryGetValue(to, out var nn) ? nn.name : "Unknown";
                targets += $"{to}({toName})";
                if (i < list.Count - 1) targets += ", ";
            }
        }
    }

    void DumpCompactStatesToLog()
    {
        if (nodes.Count == 0) return;

        List<int> ids = new List<int>(nodes.Keys);
        ids.Sort();

        StringBuilder sb = new StringBuilder();
        sb.Append("[FactoryCompact] ");

        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if (!nodes.TryGetValue(id, out var n)) continue;

            if (n.isSpawner)
            {
                sb.AppendFormat("{0}({1}):0", n.nodeId, n.name);
            }
            else
            {
                int stateCode = StateToInt(n.tunnelState);
                sb.AppendFormat("{0}({1}):{2} Q={3}/{4}",
                    n.nodeId, n.name, stateCode, n.queueCount, n.queueCapacity);
            }

            if (i < ids.Count - 1)
                sb.Append(" | ");
        }
    }

    int StateToInt(TunnelController.TunnelState s)
    {
        switch (s)
        {
            case TunnelController.TunnelState.RUN: return 0;
            case TunnelController.TunnelState.HALF_HOLD: return 1;
            case TunnelController.TunnelState.HOLD: return 2;
            case TunnelController.TunnelState.FAULT: return 3;
        }
        return -1;
    }

    public NodeData GetNode(int nodeId)
    {
        nodes.TryGetValue(nodeId, out var n);
        return n;
    }

    public List<int> GetNeighbors(int nodeId)
    {
        if (adjacency.TryGetValue(nodeId, out var list))
            return list;

        return new List<int>();
    }
}