using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// 공장 전체의 Spawner / Tunnel을 스캔해서
/// - nodeId 기준으로 상태(State, Q, Capacity)를 모으고
/// - nodeId 기준 그래프(인접 리스트)를 만든다.
/// + Assembly Reward 계산 기능 포함 (선형 흐름 가정, 경로 확률 = 1)
/// + Reward Weights (AR, QD, EC) 인스펙터에 표시
/// </summary>
public class FactoryEnvManager : MonoBehaviour
{
    [Header("Simulation Speed Control")]
    [Tooltip("시뮬레이션 시간 배속 (1 = 실시간)")]
    public float simulationTimeScale = 1f;

    // ==== Singleton ====
    public static FactoryEnvManager Instance { get; private set; }

    [Header("Scene References (비워두면 자동 찾기)")]
    public ProductSpawner[] spawners;
    public TunnelController[] tunnels;

    [Header("Reward Weights")]
    [Tooltip("Weight for Assembly Reward (AR)")]
    public float w1_AR = 1f;
    [Tooltip("Weight for Queue Drain Reward (QD) – currently 0")]
    public float w2_QD = 0f;
    [Tooltip("Weight for Energy Consumption Reward (EC) – currently 0")]
    public float w3_EC = 0f;

    // nodeId -> NodeData
    private Dictionary<int, NodeData> nodes = new Dictionary<int, NodeData>();

    // nodeId -> 나가는 child nodeId 리스트 (그래프 인접 리스트)
    private Dictionary<int, List<int>> adjacency = new Dictionary<int, List<int>>();

    // 역방향 인접 리스트 (child -> parents) – Assembly Reward 계산용
    private Dictionary<int, List<int>> reverseAdjacency = new Dictionary<int, List<int>>();

    // 에지별 분기 확률 (from, to) -> probability
    private Dictionary<(int from, int to), float> edgeProbabilities = new Dictionary<(int, int), float>();

    // 초기 스포너 생산율 (S_k(0)) – Assembly Reward 계산용
    private Dictionary<int, float> spawnerInitialRates = new Dictionary<int, float>();

    // 마지막 스냅샷 시간 – 생산율 계산용
    private float lastSnapshotTime;

    // ==== Global throughput counter (incremented by ReturnToPoolOnFinish) ====
    private int _globalExitCount = 0;

    [System.Serializable]
    public class NodeData
    {
        public int nodeId;
        public string name;

        public bool isSpawner;
        public ProductSpawner spawner;        // isSpawner == true 일 때
        public TunnelController tunnel;       // isSpawner == false 일 때

        // Tunnel인 경우에만 유효
        public TunnelController.TunnelState tunnelState;
        public int queueCount;
        public int queueCapacity;
    }

    // 외부에서 읽기용
    public IReadOnlyDictionary<int, NodeData> Nodes => nodes;
    public IReadOnlyDictionary<int, List<int>> Adjacency => adjacency;

    [Header("Debug 옵션")]
    [Tooltip("Awake 시 한 번 그래프 구조를 로그로 출력")]
    public bool debugLogOnBuild = true;

    [Header("Compact State Debug (한 줄 요약 로그)")]
    [Tooltip("true면 일정 주기로 전체 노드 상태를 한 줄로 출력")]
    public bool debugCompactState = false;
    [Tooltip("Compact 상태 로그 주기(초)")]
    public float debugCompactInterval = 1f;
    private float _nextCompactLogTime = 0f;

    // spawner rates tracking (kept for state)
    private Dictionary<ProductSpawner, int> _prevSpawnerCounts = new Dictionary<ProductSpawner, int>();

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[FactoryEnvManager] 이미 인스턴스가 존재해서 제거합니다.");
            Destroy(this);
            return;
        }
        Instance = this;

        if (spawners == null || spawners.Length == 0)
            spawners = FindObjectsOfType<ProductSpawner>();

        if (tunnels == null || tunnels.Length == 0)
            tunnels = FindObjectsOfType<TunnelController>();

        BuildNodeIndex();
        BuildGraphEdges();                     // 순방향 그래프 + edgeProbabilities 구축
        BuildReverseAdjacency();                // Assembly Reward용 역방향 그래프
        InitializeSpawnerCounts();
        InitSpawnerRates();                     // 초기 생산율 저장
        lastSnapshotTime = Time.time;

        if (debugLogOnBuild)
            DumpGraphToLog();

        _nextCompactLogTime = Time.time + debugCompactInterval;
        ApplyTimeScale();
    }

    void ApplyTimeScale()
    {
        Time.timeScale = simulationTimeScale;
        Time.fixedDeltaTime = 0.02f * Time.timeScale;
        Debug.Log($"[TimeScale] timeScale={Time.timeScale}, fixedDeltaTime={Time.fixedDeltaTime}");
    }

    void Update()
    {
        UpdateNodeStates();

        if (debugCompactState && Time.time >= _nextCompactLogTime)
        {
            DumpCompactStatesToLog();
            _nextCompactLogTime = Time.time + Mathf.Max(0.1f, debugCompactInterval);
        }
    }

    // ------------------------------------------------------------------
    // 초기화 관련 메서드
    // ------------------------------------------------------------------
    void InitializeSpawnerCounts()
    {
        if (spawners == null) return;
        foreach (var sp in spawners)
        {
            if (sp != null)
                _prevSpawnerCounts[sp] = sp.totalSpawnedCount;
        }
    }

    void InitSpawnerRates()
    {
        spawnerInitialRates.Clear();
        foreach (var sp in spawners)
        {
            if (sp != null)
            {
                // spawnInterval이 초 단위라고 가정
                float rate = sp.spawnInterval > 0 ? 1f / sp.spawnInterval : 0f;
                spawnerInitialRates[sp.nodeId] = rate;
            }
        }
    }

    // ------------------------------------------------------------------
    // 생산율 관련 (초당)
    // ------------------------------------------------------------------
    public float[] GetCurrentSpawnerRates()
    {
        if (spawners == null || spawners.Length == 0)
            return new float[0];

        float[] rates = new float[spawners.Length];
        float now = Time.time;
        float deltaTime = now - lastSnapshotTime;
        if (deltaTime <= 0f) deltaTime = 0.001f;

        for (int i = 0; i < spawners.Length; i++)
        {
            var sp = spawners[i];
            if (sp == null) continue;

            int current = sp.totalSpawnedCount;
            int prev;
            if (!_prevSpawnerCounts.TryGetValue(sp, out prev))
                prev = current;

            int delta = current - prev;
            rates[i] = delta / deltaTime;   // products per second

            _prevSpawnerCounts[sp] = current;
        }

        lastSnapshotTime = now;
        return rates;
    }

    public float GetCurrentSpawnerRate(int spawnerId)
    {
        if (!nodes.TryGetValue(spawnerId, out var node) || !node.isSpawner)
            return 0f;
        var sp = node.spawner;
        int current = sp.totalSpawnedCount;
        int prev = _prevSpawnerCounts.ContainsKey(sp) ? _prevSpawnerCounts[sp] : current;
        float deltaTime = Time.time - lastSnapshotTime;
        float rate = (current - prev) / (deltaTime > 0 ? deltaTime : 0.001f);
        Debug.Log($"[Rate] Spawner {spawnerId}: current={current}, prev={prev}, deltaTime={deltaTime:F3} -> rate={rate:F3}");
        return rate;
    }

    public float GetInitialSpawnerRate(int spawnerId)
    {
        if (spawnerInitialRates.TryGetValue(spawnerId, out float rate))
            return rate;
        return 0f;
    }

    // ------------------------------------------------------------------
    // 그래프 구축 (순방향 + 에지 확률)
    // ------------------------------------------------------------------
    void BuildGraphEdges()
    {
        adjacency.Clear();
        edgeProbabilities.Clear();

        // Spawner edges (firstTunnels + branchProbabilities)
        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;
                int fromId = sp.nodeId;
                if (fromId < 0 || !nodes.ContainsKey(fromId)) continue;

                if (!adjacency.TryGetValue(fromId, out var list))
                {
                    list = new List<int>();
                    adjacency.Add(fromId, list);
                }

                if (sp.firstTunnels != null)
                {
                    for (int i = 0; i < sp.firstTunnels.Length; i++)
                    {
                        var t = sp.firstTunnels[i];
                        if (t == null) continue;
                        int toId = t.nodeId;
                        if (toId < 0 || !nodes.ContainsKey(toId)) continue;
                        if (!list.Contains(toId))
                            list.Add(toId);

                        // Store probability (default 1 if array missing or index out of range)
                        float prob = (sp.branchProbabilities != null && i < sp.branchProbabilities.Length) ? sp.branchProbabilities[i] : 1f;
                        edgeProbabilities[(fromId, toId)] = prob;
                    }
                }
            }
        }

        // Tunnel edges (nextTunnelsForGraph + branchProbabilities)
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;
                int fromId = t.nodeId;
                if (fromId < 0 || !nodes.ContainsKey(fromId)) continue;

                if (!adjacency.TryGetValue(fromId, out var list))
                {
                    list = new List<int>();
                    adjacency.Add(fromId, list);
                }

                var next = t.nextTunnelsForGraph;
                if (next != null)
                {
                    for (int i = 0; i < next.Length; i++)
                    {
                        var child = next[i];
                        if (child == null) continue;
                        int toId = child.nodeId;
                        if (toId < 0 || !nodes.ContainsKey(toId)) continue;
                        if (!list.Contains(toId))
                            list.Add(toId);

                        // Store probability (default 1 if array missing or index out of range)
                        float prob = (t.branchProbabilities != null && i < t.branchProbabilities.Length) ? t.branchProbabilities[i] : 1f;
                        edgeProbabilities[(fromId, toId)] = prob;
                    }
                }
            }
        }
    }

    void BuildReverseAdjacency()
    {
        reverseAdjacency.Clear();
        foreach (var kv in adjacency)
        {
            int from = kv.Key;
            foreach (int to in kv.Value)
            {
                if (!reverseAdjacency.ContainsKey(to))
                    reverseAdjacency[to] = new List<int>();
                if (!reverseAdjacency[to].Contains(from))
                    reverseAdjacency[to].Add(from);
            }
        }
    }

    // ------------------------------------------------------------------
    // 경로 확률 계산
    // ------------------------------------------------------------------
    float GetEdgeProbability(int from, int to)
    {
        if (edgeProbabilities.TryGetValue((from, to), out float prob))
            return prob;
        return 1f; // fallback (should not happen if graph is consistent)
    }

    float ComputePathProbability(int sourceId, int targetId)
    {
        var visited = new HashSet<int>();
        return DFSFindPath(sourceId, targetId, visited, 1f);
    }

    float DFSFindPath(int current, int target, HashSet<int> visited, float probSoFar)
    {
        if (current == target) return probSoFar;
        visited.Add(current);
        if (adjacency.TryGetValue(current, out var children))
        {
            foreach (int child in children)
            {
                if (visited.Contains(child)) continue;
                float edgeProb = GetEdgeProbability(current, child);
                if (edgeProb <= 0) continue;
                float result = DFSFindPath(child, target, visited, probSoFar * edgeProb);
                if (result > 0) return result;
            }
        }
        return 0f;
    }

    // ------------------------------------------------------------------
    // 상위 소스 집합 찾기 (Assembly Reward용)
    // ------------------------------------------------------------------
    public HashSet<int> GetParentSources(int nodeId)
    {
        var sources = new HashSet<int>();
        var visited = new HashSet<int>();
        TraverseUpstream(nodeId, visited, sources);
        return sources;
    }

    void TraverseUpstream(int current, HashSet<int> visited, HashSet<int> sources)
    {
        if (visited.Contains(current)) return;
        visited.Add(current);
        if (nodes[current].isSpawner)
        {
            sources.Add(current);
            return;
        }
        if (reverseAdjacency.TryGetValue(current, out var parents))
        {
            foreach (int parent in parents)
                TraverseUpstream(parent, visited, sources);
        }
    }

    // ------------------------------------------------------------------
    // 마지막 고장 여부 판단 (Assembly Reward용)
    // ------------------------------------------------------------------
    public bool IsLastFault(int nodeId)
    {
        var visited = new HashSet<int>();
        return AreAllDownstreamOperational(nodeId, visited);
    }

    bool AreAllDownstreamOperational(int nodeId, HashSet<int> visited)
    {
        visited.Add(nodeId);
        if (adjacency.TryGetValue(nodeId, out var children))
        {
            foreach (int child in children)
            {
                if (visited.Contains(child)) continue;
                var childNode = nodes[child];
                if (childNode.isSpawner) continue;
                if (childNode.tunnelState == TunnelController.TunnelState.FAULT)
                    return false;
                if (!AreAllDownstreamOperational(child, visited))
                    return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Assembly Reward 계산 (경로 확률 반영)
    // ------------------------------------------------------------------
    public float ComputeAssemblyReward(int nodeId)
    {
        if (!nodes.ContainsKey(nodeId)) return 0f;
        var nodeData = nodes[nodeId];
        if (nodeData.isSpawner) return 0f;

        var sources = GetParentSources(nodeId);
        Debug.Log($"[AR] Computing reward for node {nodeId} ({nodeData.name}). Sources found: {string.Join(",", sources)}");

        float total = 0f;
        bool lastFault = IsLastFault(nodeId);
        Debug.Log($"[AR] IsLastFault = {lastFault}");

        foreach (int sourceId in sources)
        {
            float Sk0 = GetInitialSpawnerRate(sourceId);
            float Skt = GetCurrentSpawnerRate(sourceId);
            float pathProb = ComputePathProbability(sourceId, nodeId);
            Debug.Log($"[AR] Source {sourceId}: Sk0={Sk0:F3}, Skt={Skt:F3}, pathProb={pathProb:F3}");
            if (pathProb <= 0f) continue;

            float contrib;
            if (lastFault)
                contrib = (Sk0 - Skt) * pathProb;
            else
                contrib = Skt * pathProb;

            Debug.Log($"[AR] Contribution = {contrib:F3}");
            total += contrib;
        }
        Debug.Log($"[AR] Total raw AR = {total:F3}");
        return total;
    }

    // ------------------------------------------------------------------
    // 전체 보상 = w1_AR * AR + w2_QD * QD + w3_EC * EC (QD, EC는 현재 0)
    // ------------------------------------------------------------------
    public float ComputeTotalReward(int nodeId)
    {
        float ar = ComputeAssemblyReward(nodeId);
        // QD와 EC는 아직 구현되지 않았으므로 0
        return w1_AR * ar + w2_QD * 0f + w3_EC * 0f;
    }

    // ------------------------------------------------------------------
    // Product 종료 등록 (기존 코드)
    // ------------------------------------------------------------------
    public void RegisterProductExit()
    {
        _globalExitCount++;
    }

    // ------------------------------------------------------------------
    // 노드 인덱스 빌드 (기존 코드)
    // ------------------------------------------------------------------
    void BuildNodeIndex()
    {
        nodes.Clear();

        // Spawner 등록
        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;
                int id = sp.nodeId;
                if (id < 0)
                {
                    Debug.LogWarning($"[FactoryEnvManager] Spawner '{sp.name}' 의 nodeId가 설정되지 않음.");
                    continue;
                }
                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning($"[FactoryEnvManager] nodeId={id} 중복! (Spawner '{sp.name}')");
                    continue;
                }
                nodes.Add(id, new NodeData
                {
                    nodeId = id,
                    name = sp.name,
                    isSpawner = true,
                    spawner = sp,
                    tunnel = null,
                    tunnelState = TunnelController.TunnelState.RUN,
                    queueCount = 0,
                    queueCapacity = 0
                });
            }
        }

        // Tunnel 등록
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;
                int id = t.nodeId;
                if (id < 0)
                {
                    Debug.LogWarning($"[FactoryEnvManager] Tunnel '{t.name}' 의 nodeId가 설정되지 않음.");
                    continue;
                }
                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning($"[FactoryEnvManager] nodeId={id} 중복! (Tunnel '{t.name}')");
                    continue;
                }
                int qCount = (t.queue != null) ? t.queue.Count : 0;
                int qCap = (t.queue != null) ? t.queue.Capacity : 0;

                nodes.Add(id, new NodeData
                {
                    nodeId = id,
                    name = t.name,
                    isSpawner = false,
                    spawner = null,
                    tunnel = t,
                    tunnelState = t.State,
                    queueCount = qCount,
                    queueCapacity = qCap
                });
            }
        }
    }

    // ------------------------------------------------------------------
    // 노드 상태 갱신 (기존 코드)
    // ------------------------------------------------------------------
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
                data.tunnelState = t.State;
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

    // ===================== Debug 출력 =====================
    void DumpGraphToLog()
    {
        Debug.Log("===== FactoryEnvManager Graph Dump =====");
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
            Debug.Log($"{from}({fromName}) -> [{targets}]");
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
                sb.AppendFormat("{0}({1}):0", n.nodeId, n.name);
            else
            {
                int stateCode = StateToInt(n.tunnelState);
                sb.AppendFormat("{0}({1}):{2} Q={3}/{4}",
                    n.nodeId, n.name, stateCode, n.queueCount, n.queueCapacity);
            }
            if (i < ids.Count - 1) sb.Append(" | ");
        }
        Debug.Log(sb.ToString());
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

    // ===================== 외부 헬퍼 =====================
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