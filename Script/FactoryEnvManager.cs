using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// 공장 전체의 Spawner / Tunnel을 스캔해서
/// - nodeId 기준으로 상태(State, Q, Capacity)를 모으고
/// - nodeId 기준 그래프(인접 리스트)를 만든다.
/// + 터널의 FAULT 진입/탈출을 감지해서 고장 리워드(테스트용)를 관리한다.
/// + PDF 수식 기반의 "글로벌 보상 R_total"을 계산해 테스트 로그를 출력한다.
/// + (추가) 의사결정 시점마다 관찰 윈도우 T 동안 PL/QD/FT/BT를 샘플링해서
///         윈도우 단위 리워드를 계산할 수 있다.
/// 나중에 RL / Python 브릿지에서 이 매니저만 바라보면 됨.
/// </summary>
public class FactoryEnvManager : MonoBehaviour
{

    [Header("Simulation Speed Control")]
[Tooltip("시뮬레이션 시간 배속 (1 = 실시간)")]
public float simulationTimeScale = 1f;

    // ==== Singleton (편의용) ====
    public static FactoryEnvManager Instance { get; private set; }
    [Header("Debug / PL Logs")]
    [Tooltip("관찰 윈도우가 끝날 때 PL(T) 값을 로그로 출력할지 여부")]
    public bool debugLogWindowPl = true;

    [Header("Scene References (비워두면 자동 찾기)")]
    public ProductSpawner[] spawners;
    public TunnelController[] tunnels;
    // 즉시형 글로벌 리워드용: 지난 로그 시점의 sink throughput 저장
    private Dictionary<TunnelController, int> _prevSinkExitCountsInstant
        = new Dictionary<TunnelController, int>();

    // nodeId -> NodeData
    private Dictionary<int, NodeData> nodes = new Dictionary<int, NodeData>();

    // nodeId -> 나가는 child nodeId 리스트 (그래프 인접 리스트)
    private Dictionary<int, List<int>> adjacency = new Dictionary<int, List<int>>();

    // 각 터널의 "이전 프레임 상태"를 기억해서 Fault 진입/탈출을 감지
    private Dictionary<TunnelController, TunnelController.TunnelState> _lastTunnelStates
        = new Dictionary<TunnelController, TunnelController.TunnelState>();

    // ===== 테스트용 고장 리워드 관리 (Fault별 점수) =====
    [Header("Fault Reward (테스트용, per-tunnel)")]
    [Tooltip("고장 발생 시 부여할 최소 리워드")]
    public float minFaultReward = 1f;

    [Tooltip("고장 발생 시 부여할 최대 리워드")]
    public float maxFaultReward = 5f;

    // 고장난 터널 -> 리워드 점수
    private Dictionary<TunnelController, float> faultRewards
        = new Dictionary<TunnelController, float>();

    // ===== 글로벌 리워드 (PDF 수식 기반) =====
    //
    // 슬라이드의 개념:
    //   PL(T) : 생산량
    //   QD(T) : 큐 길이 (혼잡도)
    //   FT(T) : 고장 시간
    //   BT(T) : 라인 블로킹 시간
    //   EC(T) : 에너지
    //   RO(T) : 로봇 운용 비용
    //
    //   R_total = w1 * PL~ - w2 * QD~ - w3 * FT~ - w4 * BT~ - w5 * EC~ - w6 * RO~
    //
    // 여기서는 단순화 버전으로:
    //   - PL : sink 터널의 throughput (관찰 윈도우에서 delta count 사용 가능)
    //   - QD : 전체 큐 길이 합
    //   - FT : FAULT 터널 개수
    //   - BT : HOLD + HALF_HOLD 터널 개수
    //   - EC, RO : 지금은 0 (나중에 로봇 이동량/수리 횟수와 연결 가능)
    //
    [Header("RL Reward (Global, Next-week Formula / Test)")]
    [Tooltip("R_total을 주기적으로 로그 출력할지 여부")]
    public bool debugLogGlobalReward = true;

    [Tooltip("글로벌 리워드 로그 주기(초, 즉시형 인스턴트 리워드)")]
    public float globalRewardLogInterval = 1f;

    private float _nextGlobalRewardLogTime = 0f;

    [Header("Reward Weights (w1~w6)")]
    [Tooltip("생산량 PL~의 가중치 (좋은 항, +)")]
    public float w1_PL = 1f;

    [Tooltip("큐 변화량 |QD(st+1)-QD(st)| 의 가중치 (좋은 항, +)")]
    public float w2_QD = 1f;

    [Tooltip("블로킹 변화량 |BT(st+1)-BT(st)| 의 가중치 (좋은 항, +)")]
    public float w4_BT = 1f;

    [Tooltip("에너지 EC~의 가중치 (나쁜 항, -)")]
    public float w5_EC = 0f; // 아직 미사용이므로 0으로 시작

    [Tooltip("로봇 운용비 RO~의 가중치 (나쁜 항, -)")]
    public float w6_RO = 0f; // 아직 미사용이므로 0으로 시작

    [Header("Reward Normalizers (max 값 가정)")]
    [Tooltip("PL 정규화용 최대값 (예: 시간 T 동안 가능한 최대 생산량)")]
    public float maxPL = 1f;

    [Tooltip("QD 변화량 정규화용 최대값 (예: |QD(st+1)-QD(st)|의 최댓값 가정)")]
    public float maxQD = 10f;

    [Tooltip("BT 변화량 정규화용 최대값 (예: |BT(st+1)-BT(st)|의 최댓값 가정)")]
    public float maxBT = 5f;

    [Tooltip("EC 정규화용 최대값 (에너지)")]
    public float maxEC = 1f;

    [Tooltip("RO 정규화용 최대값 (로봇 운용비)")]
    public float maxRO = 1f;

    // 최근에 계산된 글로벌 리워드 값
    private float _lastGlobalReward = 0f;
    // 최근 관찰 윈도우에서 계산된 PL(T)와 정규화된 PL~
    private float _lastWindowPlT = 0f;
    private float _lastWindowPlNorm = 0f;
    // Add these fields to FactoryEnvManager class:
    private float _lastPlRaw = 0f;
    private float _lastWindowQdDelta = 0f;
    private float _lastWindowBtDelta = 0f;
    private float _lastInstantQdDelta = 0f;
    private float _lastInstantBtDelta = 0f;

    // === 전역 throughput 카운터 (ReturnToPoolOnFinish에서 증가시킴) ===
    private int _globalExitCount = 0;           // 지금까지 끝까지 간 Product 수
    private int _exitCountAtObsStart = 0;       // 관찰 윈도우 시작 시점의 값
    // 🔹 PL 변화량 계산용: 이전 윈도우의 PL 값을 저장
    private int _prevWindowExitCount = 0;       // 이전 윈도우에서 완료된 제품 수
    private float _prevWindowPlT = 0;             // 이전 윈도우의 PL(T) 값
    
    // 🔹 instant QD/BT 변화량 계산용: 지난 instant 로그 시점의 스냅샷
    private int _prevTotalQDInstant = 0;
    private int _prevTotalBTInstant = 0;



    /// <summary>
    /// Product가 경로를 끝까지 따라간 뒤 풀로 리턴될 때 호출되는 전역 카운터
    /// </summary>
    public void RegisterProductExit()
    {
        _globalExitCount++;
    }

    // ===== 관찰 윈도우 기반 리워드 (R_t for one decision) =====
    [Header("RL Observation Window (per decision)")]
    [Tooltip("의사결정마다 T초 동안 PL/QD/FT/BT를 관찰해 윈도우 리워드를 계산할지 여부")]
    public bool useObservationWindow = false;

    [Tooltip("관찰 윈도우 길이 T (seconds)")]
    public float observationWindow = 8f;

    bool _isObserving = false;
    float _obsEndTime;

    // (추가) s_t 스냅샷 값 저장
    private int _qdAtObsStart = 0;   // QD(s_t) = 전체 큐의 합
    private int _btAtObsStart = 0;   // BT(s_t) = HOLD/HALF_HOLD 노드 수

    // 시간 평균을 위한 누적값 (기존 필드 유지)
    float _sumQD, _sumFT, _sumBT, _sumEC, _sumRO;
    int _sampleCount;

    // sink 터널들의 시작 시점 throughput 카운트
    Dictionary<TunnelController, int> _sinkStartCounts
        = new Dictionary<TunnelController, int>();

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

    void Awake()
    {
        // Singleton 세팅
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[FactoryEnvManager] 이미 인스턴스가 존재해서 두 번째 인스턴스를 제거합니다.");
            Destroy(this);
            return;
        }
        Instance = this;

        // 씬에서 자동 스캔 (인스펙터에서 수동 지정해도 됨)
        if (spawners == null || spawners.Length == 0)
            spawners = FindObjectsOfType<ProductSpawner>();

        if (tunnels == null || tunnels.Length == 0)
            tunnels = FindObjectsOfType<TunnelController>();

        BuildNodeIndex();
        BuildGraphEdges();
        InitializeSpawnerCounts();  // <-- add this line

        if (debugLogOnBuild)
        {
            DumpGraphToLog();
        }

        _nextCompactLogTime = Time.time + debugCompactInterval;
        _nextGlobalRewardLogTime = Time.time + globalRewardLogInterval;
        ApplyTimeScale();
    }
    
    void ApplyTimeScale()
    {
        Time.timeScale = simulationTimeScale;
        Time.fixedDeltaTime = 0.02f * Time.timeScale;

        Debug.Log(
            $"[TimeScale] timeScale={Time.timeScale}, fixedDeltaTime={Time.fixedDeltaTime}"
        );
    }

    void Update()
    {
        // 매 프레임마다 상태만 갱신
        UpdateNodeStates();

        // 한 줄 compact 로그
        if (debugCompactState && Time.time >= _nextCompactLogTime)
        {
            DumpCompactStatesToLog();
            _nextCompactLogTime = Time.time + Mathf.Max(0.1f, debugCompactInterval);
        }

        // === 관찰 윈도우 T 처리 (의사결정 기반 리워드) ===
        if (useObservationWindow && _isObserving)
        {
            SampleForObservation();

            if (Time.time >= _obsEndTime)
            {
                FinishObservationAndComputeReward();
            }
        }

        // === PDF 수식 기반 "즉시형" 글로벌 리워드 로그 (선택) ===
        if (debugLogGlobalReward && Time.time >= _nextGlobalRewardLogTime)
        {
            float plT, qdT, btT, ecT, roT;
            float plN, qdN, btN, ecN, roN;

            float r = ComputeGlobalReward(
                out plT, out qdT, out btT, out ecT, out roT,
                out plN, out qdN, out btN, out ecN, out roN
            );

            _lastGlobalReward = r;
            // 🔹 관찰 윈도우 기준 PL(T), PL~ 저장
            _lastWindowPlT = plT;
            _lastWindowPlNorm = plN;
            float termPL = + w1_PL * plN;
            float termQD = + w2_QD * qdN;
            float termBT = + w4_BT * btN;
            float termEC = - w5_EC * ecN;
            float termRO = - w6_RO * roN;

            Debug.Log(
                $"[FactoryReward(instant)] R={r:F3} = " +
                $"{termPL:F3}(PL) + {termQD:F3}(dQD) + {termBT:F3}(dBT) + {termEC:F3}(EC) + {termRO:F3}(RO)\n" +
                $"  raw:   PL={plT:F0}, dQD=|Δ|={qdT:F0}, dBT=|Δ|={btT:F0}, EC={ecT:F2}, RO={roT:F2}\n" +
                $"  norm:  PL~={plN:F3}, dQD~={qdN:F3}, dBT~={btN:F3}, EC~={ecN:F3}, RO~={roN:F3}\n" +
                $"  w:     w1={w1_PL:F2}, w2={w2_QD:F2}, w4={w4_BT:F2}, w5={w5_EC:F2}, w6={w6_RO:F2}"
            );



            _nextGlobalRewardLogTime = Time.time + Mathf.Max(0.1f, globalRewardLogInterval);
        }
    }

    //// spawn info ////
    // At the top with other dictionaries
    private Dictionary<ProductSpawner, int> _prevSpawnerCounts = new Dictionary<ProductSpawner, int>();

    // Call this after spawners are assigned (e.g., in Start or after BuildNodeIndex)
    void InitializeSpawnerCounts()
    {
        if (spawners == null) return;
        foreach (var sp in spawners)
        {
            if (sp != null)
                _prevSpawnerCounts[sp] = sp.totalSpawnedCount;
        }
    }

    // Public method called by DqnAgent to get rates for the current snapshot
    public float[] GetCurrentSpawnerRates()
    {
        if (spawners == null || spawners.Length == 0)
            return new float[0];

        float[] rates = new float[spawners.Length];
        for (int i = 0; i < spawners.Length; i++)
        {
            var sp = spawners[i];
            if (sp == null) continue;

            int current = sp.totalSpawnedCount;
            int prev;
            if (!_prevSpawnerCounts.TryGetValue(sp, out prev))
                prev = current; // first time

            int delta = current - prev;
            rates[i] = delta; // products spawned since last snapshot

            // update for next time
            _prevSpawnerCounts[sp] = current;
        }
        return rates;
    }

    /// finish spawn info /////

    // ===================== 관찰 윈도우 API =====================

    /// <summary>
    /// 로봇이 "다음 수리 대상"을 결정하는 시점에 한번 호출해주면 됨.
    /// observationWindow 동안 QD/FT/BT를 샘플링하고,
    /// sink 터널의 throughput delta로 PL(T)를 계산한다.
    /// </summary>
    public void BeginRewardObservation()
    {
        if (!useObservationWindow)
            return;

        _isObserving = true;
        _obsEndTime = Time.time + observationWindow;

        _sumQD = _sumFT = _sumBT = _sumEC = _sumRO = 0f;
        _sampleCount = 0;
        _sinkStartCounts.Clear();

        // 🔸 관찰 시작 시점의 전역 throughput 카운트 저장
        _exitCountAtObsStart = _globalExitCount;

        // 🔸 s_t 시점 스냅샷(QD, BT) 저장
        int startQD = 0;
        int startBT = 0;
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                if (t.queue != null)
                    startQD += t.queue.Count;

                if (t.IsHold || t.IsHalfHold)
                    startBT++;
            }
        }
        _qdAtObsStart = startQD;
        _btAtObsStart = startBT;

        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                // TunnelController에 isSink, totalExitedCount가 있다고 가정
                if (t.isSink)
                    _sinkStartCounts[t] = t.totalExitedCount;
            }
        }
    }

    /// <summary>
    /// 관찰 윈도우 중 매 프레임 호출되어,
    /// QD/FT/BT를 time-average를 위해 누적한다.
    /// </summary>
    void SampleForObservation()
    {
        if (tunnels == null) return;

        int totalQD = 0;
        int faultCount = 0;
        int blockCount = 0;

        foreach (var t in tunnels)
        {
            if (t == null) continue;

            if (t.queue != null)
                totalQD += t.queue.Count;

            if (t.IsFault)
                faultCount++;

            if (t.IsHold || t.IsHalfHold)
                blockCount++;
        }

        _sumQD += totalQD;
        _sumFT += faultCount;
        _sumBT += blockCount;
        // EC/RO는 아직 0으로 둔 상태
        _sampleCount++;
    }

    /// <summary>
    /// 관찰 윈도우가 끝났을 때 호출.
    /// 평균 QD/FT/BT와 sink throughput delta로 PL(T)을 계산하고,
    /// 글로벌 리워드를 한 번 로그로 출력한다.
    /// </summary>
    void FinishObservationAndComputeReward()
    {
        _isObserving = false;

        if (_sampleCount <= 0)
            return;

        // 관찰 윈도우 동안의 시간 평균 (기존 로그/유지 목적)
        float avgQD = _sumQD / _sampleCount;
        float avgFT = _sumFT / _sampleCount;
        float avgBT = _sumBT / _sampleCount;
        float ecT = 0f;
        float roT = 0f;

        // 🔸 PL(T): 관찰 윈도우 동안 "끝까지 간" 제품 수 (전역 카운터 delta)
        int deltaExit = Mathf.Max(0, _globalExitCount - _exitCountAtObsStart);
        int plCurrentWindow = deltaExit; // 현재 윈도우의 PL
        
        // 🔸 NEW: 현재 윈도우 PL에서 이전 윈도우 PL을 뺀 값 사용
        float plDelta = plCurrentWindow - _prevWindowPlT;
        
        // 🔸 다음 윈도우를 위해 현재 윈도우 값을 저장


        // 🔸 s_{t+1} 시점(관찰 종료 시점) 스냅샷(QD, BT)
        int endQD = 0;
        int endBT = 0;
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                if (t.queue != null)
                    endQD += t.queue.Count;

                if (t.IsHold || t.IsHalfHold)
                    endBT++;
            }
        }

        // ✅ 새 정의: QD, BT는 "절대값 변화량"
        float qdDeltaAbs = (_qdAtObsStart - endQD);
        float btDeltaAbs = (_btAtObsStart - endBT);

        // Store for later retrieval
        _lastPlRaw = plDelta;
        _lastWindowQdDelta = qdDeltaAbs;
        _lastWindowBtDelta = btDeltaAbs;

        // In FinishObservationAndComputeReward(), after calculating:
        Debug.Log($"[WindowRaw] startQD={_qdAtObsStart}, endQD={endQD}, delta={qdDeltaAbs}");
        Debug.Log($"[WindowRaw] startBT={_btAtObsStart}, endBT={endBT}, delta={btDeltaAbs}");

        float plN, qdN, btN, ecN, roN;
        
        // 🔸 NEW: PL 변화량(plDelta)을 사용하여 리워드 계산
        float r = ComputeGlobalRewardFromValues(
            plDelta, qdDeltaAbs, btDeltaAbs, ecT, roT,
            out plN, out qdN, out btN, out ecN, out roN
        );

        _lastGlobalReward = r;
        _lastWindowPlT = plDelta;        // 저장할 때는 변화량 저장
        _lastWindowPlNorm = plN;

        if (debugLogWindowPl)
        {
            Debug.Log(
                $"[FactoryReward(window-metrics)] T={observationWindow:F1}s\n" +
                $"  PL: exitCount { _exitCountAtObsStart } -> { _globalExitCount }  => PL(current_window)={plCurrentWindow:F0}\n" +
                $"  PL_delta: current({plCurrentWindow:F0}) - prev({_prevWindowPlT:F0}) = {plDelta:F0}\n" +
                $"  QD: totalQ  {_qdAtObsStart} -> {endQD}  => dQD=|Δ|={qdDeltaAbs:F0}\n" +
                $"  BT: blocked {_btAtObsStart} -> {endBT}  => dBT=|Δ|={btDeltaAbs:F0}\n" +
                $"  (avg for reference) QD_avg={avgQD:F2}, BT_avg={avgBT:F2}"
            );

        }
        _prevWindowPlT = plCurrentWindow;
        if (debugLogGlobalReward)
        {
            Debug.Log(
                $"[FactoryReward(window)] R={r:F3} | " +
                $"PL_delta={plDelta:F2}, dQD={qdDeltaAbs:F2}, dBT={btDeltaAbs:F2} | " +
                $"PL~={plN:F2}, dQD~={qdN:F2}, dBT~={btN:F2}, EC~={ecN:F2}, RO~={roN:F2}"
            );
        }
    }

    // ===================== 노드 인덱스 =====================

    void BuildNodeIndex()
    {
        nodes.Clear();
        _lastTunnelStates.Clear();
        faultRewards.Clear();

        // 1) Spawner → 노드 등록
        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;

                int id = sp.nodeId;  // ProductSpawner에 public int nodeId
                if (id < 0)
                {
                    Debug.LogWarning($"[FactoryEnvManager] Spawner '{sp.name}' 의 nodeId가 설정되지 않음 (<0). 그래프에서 제외.");
                    continue;
                }

                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning($"[FactoryEnvManager] nodeId={id} 중복! (Spawner '{sp.name}')");
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

        // 2) Tunnel → 노드 등록
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                int id = t.nodeId;   // TunnelController에 public int nodeId
                if (id < 0)
                {
                    Debug.LogWarning($"[FactoryEnvManager] Tunnel '{t.name}' 의 nodeId가 설정되지 않음 (<0). 그래프에서 제외.");
                    continue;
                }

                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning($"[FactoryEnvManager] nodeId={id} 중복! (Tunnel '{t.name}')");
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

                // 터널의 초기 상태를 "이전 상태" 딕셔너리에 저장
                _lastTunnelStates[t] = t.State;
            }
        }
    }

    // ===================== 그래프 간선 빌드 =====================

    void BuildGraphEdges()
    {
        adjacency.Clear();

        // 1) Spawner: nodeId -> firstTunnels[].nodeId
        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;
                int fromId = sp.nodeId;
                if (fromId < 0) continue;
                if (!nodes.ContainsKey(fromId)) continue;

                if (!adjacency.TryGetValue(fromId, out var list))
                {
                    list = new List<int>();
                    adjacency.Add(fromId, list);
                }

                if (sp.firstTunnels != null)
                {
                    foreach (var t in sp.firstTunnels)
                    {
                        if (t == null) continue;
                        int toId = t.nodeId;
                        if (toId < 0) continue;
                        if (!nodes.ContainsKey(toId)) continue;

                        if (!list.Contains(toId))
                            list.Add(toId);
                    }
                }
            }
        }

        // 2) Tunnel: nodeId -> nextTunnelsForGraph[]
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;
                int fromId = t.nodeId;
                if (fromId < 0) continue;
                if (!nodes.ContainsKey(fromId)) continue;

                if (!adjacency.TryGetValue(fromId, out var list))
                {
                    list = new List<int>();
                    adjacency.Add(fromId, list);
                }

                var next = t.nextTunnelsForGraph;  // TunnelController에 public 필드
                if (next == null) continue;

                foreach (var child in next)
                {
                    if (child == null) continue;
                    int toId = child.nodeId;
                    if (toId < 0) continue;
                    if (!nodes.ContainsKey(toId)) continue;

                        if (!list.Contains(toId))
                            list.Add(toId);
                }
            }
        }
    }

    // ===================== 상태 갱신 + 고장 리워드 이벤트 =====================

    void UpdateNodeStates()
    {
        // Tunnel 상태/큐 정보만 주기적으로 업데이트
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

                // 이전 상태가 있으면 Fault 진입/탈출 감지
                if (_lastTunnelStates.TryGetValue(t, out var prevState))
                {
                    // 비-FAULT → FAULT : 고장 발생
                    if (prevState != TunnelController.TunnelState.FAULT &&
                        currentState == TunnelController.TunnelState.FAULT)
                    {
                        OnTunnelFailed(t);
                    }
                    // FAULT → 비-FAULT : 수리 완료
                    else if (prevState == TunnelController.TunnelState.FAULT &&
                             currentState != TunnelController.TunnelState.FAULT)
                    {
                        OnTunnelRepaired(t);
                    }
                }

                // 현재 상태를 "이전 상태"로 갱신
                _lastTunnelStates[t] = currentState;

                // NodeData 갱신
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

    // ----- Fault Reward 내부 처리 (per-tunnel) -----

    void OnTunnelFailed(TunnelController t)
    {
        // 테스트용: 고장마다 랜덤 리워드 부여
        float reward = Random.Range(minFaultReward, maxFaultReward);
        faultRewards[t] = reward;
        // Debug.Log($"[FactoryEnvManager] Tunnel FAILED '{t.name}', reward={reward}");
    }

    void OnTunnelRepaired(TunnelController t)
    {
        if (faultRewards.ContainsKey(t))
        {
            faultRewards.Remove(t);
            // Debug.Log($"[FactoryEnvManager] Tunnel REPAIRED '{t.name}', remove reward entry");
        }
    }

    /// <summary>
    /// 현재 고장난 터널들 중에서 리워드가 가장 큰 터널을 반환.
    /// 없으면 null.
    /// (테스트용 정책: 리워드가 클수록 먼저 수리하러 감)
    /// </summary>
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

    // ----- 글로벌 리워드 계산 (수정된 버전: FT 제거, QD/BT는 변화량 |Δ| 보상) -----

    /// <summary>
    /// 현재 상태에서 PL(T), QD(T), BT(T), EC(T), RO(T)를
    /// 단순하게 추정한다.
    /// </summary>
    void ComputeRawMetrics(
        out float PL, out float QD,
        out float BT,
        out float EC, out float RO)
    {
        // 0으로 초기화
        PL = 0f;
        QD = 0f;
        BT = 0f;
        EC = 0f;
        RO = 0f;

        // 1) QD / BT : 전체 터널 스냅샷 기준 (현재값)
        int currentQD = 0;
        int currentBT = 0;

        foreach (var kv in nodes)
        {
            var n = kv.Value;
            if (n.isSpawner) continue;

            currentQD += n.queueCount;

            switch (n.tunnelState)
            {
                case TunnelController.TunnelState.HOLD:
                case TunnelController.TunnelState.HALF_HOLD:
                    currentBT += 1;
                    break;
            }
        }

        // ✅ 새 정의(즉시형): s_t(이전 로그 시점) 대비 s_{t+1}(현재) 변화량의 절대값
        QD = (_prevTotalQDInstant - currentQD);
        BT = (_prevTotalBTInstant - currentBT);
        // Store for retrieval
        _lastInstantQdDelta = QD;
        _lastInstantBtDelta = BT;

        _prevTotalQDInstant = currentQD;
        _prevTotalBTInstant = currentBT;

        // 2) PL : 지난 instant 로그 이후 끝까지 간 제품 수 (전역 카운터 delta)
        // 🔸 NEW: Instant 모드에서도 변화량 개념 사용 (이전 instant 대비)
        int deltaExit = _globalExitCount - _prevWindowExitCount;
        if (deltaExit > 0)
            PL = deltaExit;
        else
            PL = 0f;

        _prevWindowExitCount = _globalExitCount;

        // EC, RO는 나중에 로봇 이동/수리 횟수 붙이고 싶을 때 채우면 됨.
    }

    /// <summary>
    /// 주어진 PL/QD/BT/EC/RO 값으로부터
    /// 정규화된 항들을 계산하고,
    /// R_total = + w1*PL~ + w2*QD~ + w4*BT~ - w5*EC~ - w6*RO~
    /// (관찰 윈도우 / 즉시형 모두 공용으로 사용)
    /// </summary>
    public float ComputeGlobalRewardFromValues(
        float PL, float QD, float BT, float EC, float RO,
        out float PL_norm, out float QD_norm,
        out float BT_norm,
        out float EC_norm, out float RO_norm)
    {
        // 🔸 NEW: PL 정규화는 -1~1 범위로 변경 (변화량이 음수일 수 있으므로)
        // Optional: Clip extreme negative values
        //QD_norm = Mathf.Clamp(QD_norm, -1f, 1f);  // Limit to [-1, 1]
        //BT_norm = Mathf.Clamp(BT_norm, -1f, 1f);  // Limit to [-1, 1]
        PL_norm = (maxPL > 0f) ? Mathf.Clamp(PL / maxPL, -1f, 1f ) : 0f;
        QD_norm = (maxQD > 0f) ? Mathf.Clamp(QD / maxQD, -1f, 1f ) : 0f;
        BT_norm = (maxBT > 0f) ? Mathf.Clamp(BT / maxBT, -1f, 1f ) : 0f;
        EC_norm = (maxEC > 0f) ? Mathf.Clamp01(EC / maxEC) : 0f;
        RO_norm = (maxRO > 0f) ? Mathf.Clamp01(RO / maxRO) : 0f;

        float reward =
            + w1_PL * PL_norm
            + w2_QD * QD_norm
            + w4_BT * BT_norm
            - w5_EC * EC_norm
            - w6_RO * RO_norm;

        return reward;
    }

    /// <summary>
    /// "현재 시점"의 상태로부터 글로벌 리워드를 계산.
    /// (기존 즉시형 로그용, 관찰 윈도우가 아니라 그냥 스냅샷 기준)
    /// </summary>
    public float ComputeGlobalReward(
        out float PL, out float QD,
        out float BT,
        out float EC, out float RO,
        out float PL_norm, out float QD_norm,
        out float BT_norm,
        out float EC_norm, out float RO_norm)
    {
        ComputeRawMetrics(out PL, out QD, out BT, out EC, out RO);
        return ComputeGlobalRewardFromValues(
            PL, QD, BT, EC, RO,
            out PL_norm, out QD_norm,
            out BT_norm,
            out EC_norm, out RO_norm
        );
    }

    /// <summary>
    /// 최근에 계산된 글로벌 리워드 값을 읽고 싶을 때 사용.
    /// (즉시형/윈도우형 둘 중 마지막으로 계산된 값)
    /// </summary>
    public float GetLastGlobalReward()
    {
        return _lastGlobalReward;
    }
    /// <summary>
    /// 최근 관찰 윈도우의 PL(T) 원값
    /// </summary>
    public float GetLastWindowPl()
    {
        return _lastWindowPlT;
    }

    /// <summary>
    /// 최근 관찰 윈도우의 정규화된 PL~ 값
    /// </summary>
    public float GetLastWindowPlNorm()
    {
        return _lastWindowPlNorm;
    }

    public float GetLastWindowQdDelta()
    {
        return _lastWindowQdDelta;
    }

    public float GetLastWindowBtDelta()
    {
        return _lastWindowBtDelta;
    }

    public float GetCurrentQdDelta()
    {
        return _lastInstantQdDelta;
    }

    public float GetCurrentBtDelta()
    {
        return _lastInstantBtDelta;
    }
    public float GetPlRaw()
    {
        return _lastPlRaw;
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

    /// <summary>
    /// 한 줄로 전체 노드 상태를 compact하게 출력
    /// 예: [FactoryCompact] 0(Spawner_S1):0 | 1(Tunnel_A):2 Q=3/5 | ...
    /// 상태 코드: RUN=0, HALF_HOLD=1, HOLD=2, FAULT=3
    /// </summary>
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
                // Spawner는 상태 코드 0으로 통일
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

    // ===================== 외부에서 쓸 수 있는 헬퍼 =====================

    /// <summary>
    /// 특정 nodeId의 상태를 얻는다. 존재하지 않으면 null 반환.
    /// </summary>
    public NodeData GetNode(int nodeId)
    {
        nodes.TryGetValue(nodeId, out var n);
        return n;
    }

    /// <summary>
    /// 특정 nodeId에서 나가는 child nodeId 리스트를 얻는다. 없으면 빈 리스트 반환.
    /// </summary>
    public List<int> GetNeighbors(int nodeId)
    {
        if (adjacency.TryGetValue(nodeId, out var list))
            return list;
        return new List<int>();
    }
}