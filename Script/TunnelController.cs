using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using URandom = UnityEngine.Random;

[DisallowMultipleComponent]
public class TunnelController : MonoBehaviour
{
    // ===== Graph Node Id =====
    [Header("Graph Node Id")]
    [Tooltip("FactoryEnvManager에서 쓰는 노드 번호 (Spawner와 연속되게 부여)")]
    public int nodeId = -1;

    [Header("Node Selection")]
    [Tooltip("Check to enable node filtering. Uncheck to use all nodes.")]
    public bool useNodeFilter = false;

    [Tooltip("Array of toggles for each node (index = nodeId).")]
    public bool[] nodeToggleArray = new bool[34];


    [Header("Graph Next Nodes (for FactoryEnvManager)")]
    [Tooltip("그래프/경로 계산용으로, 이 터널 다음에 갈 수 있는 터널들 (Spawner의 firstTunnels처럼 수동 지정)")]
    public TunnelController[] nextTunnelsForGraph;

    // ===== Sink / Throughput (for RL) =====
    [Header("Sink / Throughput (for RL)")]
    [Tooltip("이 터널이 최종 배출 지점(라인 끝)인지 여부 (PL(T) 계산용)")]
    public bool isSink = false;

    [Tooltip("이 터널에서 지금까지 배출된 제품 개수 (RL throughput 계산용)")]
    public int totalExitedCount = 0;

        // Add this in the [Header("Upstream Spawners")] section or create a new one:
    [Header("Upstream Spawners (for speed adjustment)")]
    [Tooltip("Spawners that should slow down when this tunnel is in FAULT")]
    public ProductSpawner[] upstreamSpawners;

    [Tooltip("Multiplier for spawn speed when tunnel is in FAULT (0.5 = half speed)")]
    [Range(0.1f, 1f)]
    public float faultSpeedMultiplier = 0.5f;

    [Tooltip("Should upstream spawners slow down when this tunnel is in FAULT?")]
    public bool slowUpstreamWhenFault = true;

    // Add this dictionary to track which tunnels are affecting which spawners
    private Dictionary<ProductSpawner, float> originalSpawnMultipliers = new Dictionary<ProductSpawner, float>();

    // ===== 4-State =====
    // RUN : 정상
    // HALF_HOLD : 일부 downstream만 막힌 상태(부분 정체)
    // HOLD : 완전 정체 (전달/스폰 중단)
    // FAULT : 고장 상태
    public enum TunnelState { RUN, HALF_HOLD, HOLD, FAULT }

    // ===== Failure Model =====
    public enum FailureModel
    {
        Constant,   // 평균 개수 기반 고정 hazard (지수/기하 형태)
        Gaussian,   // "몇 개쯤에서 잘 고장나는지"를 가우시안으로 모델링
        Exponential // 평균 개수 기반 지수 분포 (개수가 늘수록 고장 수명 랜덤)
    }

    [Header("State (4-state)")]
    [SerializeField] private TunnelState state = TunnelState.RUN;
    public TunnelState State => state;

    public bool IsRun      => state == TunnelState.RUN;
    public bool IsHalfHold => state == TunnelState.HALF_HOLD;
    public bool IsHold     => state == TunnelState.HOLD;
    public bool IsFault    => state == TunnelState.FAULT;

    // ===== Failure Model Config (Items-based) =====
    [Header("Failure Model (터널 고장 모델 선택)")]
    [SerializeField] private FailureModel failureModel = FailureModel.Constant;

    [Tooltip("true면 제품 개수 기반 확률 고장을 사용, false면 시나리오 등에서 외부 ForceFail만 사용")]
    public bool useItemFailure = true;

    [Header("Failure - Constant (Items-based)")]
    [Tooltip("평균 몇 개의 제품을 처리하면 고장날지 (지수/기하 분포 기반)")]
    public float constantMeanItemsToFailure = 200f;

    [Header("Failure - Gaussian (Items-based)")]
    [Tooltip("평균 몇 개의 제품을 처리했을 때 고장이 잘 나는지 (μ)")]
    public float gaussianMeanItems = 200f;

    [Tooltip("고장 발생 제품 개수 분포의 표준편차 (σ)")]
    public float gaussianStdItems = 50f;

    [Tooltip("Gaussian 모델에서 최소 보장 제품 개수 (너무 작은 값 방지)")]
    public int gaussianMinItems = 10;

    [Header("Failure - Exponential (Items-based)")]
    [Tooltip("평균 몇 개의 제품을 처리하면 고장날지 (지수 분포 평균 M)")]
    public float exponentialMeanItems = 300f;

    [Tooltip("Exponential 모델에서 최소 보장 제품 개수")]
    public int exponentialMinItems = 5;

    [Header("Failure Items Tracking (Debug)")]
    [Tooltip("마지막 고장/수리 이후 처리한 제품 개수")]
    [SerializeField] private int itemsSinceLastFailure = 0;

    [Tooltip("이번 사이클에서 고장이 발생할 목표 제품 개수 (itemsSinceLastFailure가 이 값에 도달하면 FAULT)")]
    [SerializeField] private int nextFailureAtItem = int.MaxValue;

    [Header("Dispatch Tempo (optional)")]
    [Tooltip("한 번에 몇 개까지 내보낼지 (큐에서 최대 pop 개수)")]
    public int dispatchPerTick = 1;

    [Tooltip("두 방출 사이 기본 간격(초). spawn interval처럼 동작.")]
    public float dispatchCooldown = 0.5f;

    [Header("Wiring")]
    public QueueZone queue;
    public Transform afterTunnelPoint;

    [Header("Auto Repair (Fixed Delay, Test)")]
    [Tooltip("체크 시 FAULT 진입 후 fixedRepairDelay 초 뒤 자동 수리, 해제 시 자동 수리 없음")]
    public bool autoRepairFixedDelay = true;
    public float fixedRepairDelay = 15f;
    public KeyCode debugRepairKey = KeyCode.R;

    // ===== Visual =====
    [Header("Status Visual")]
    [SerializeField] private Renderer statusRenderer;
    [SerializeField] private string colorProperty = "_Color";
    [SerializeField] private Color okColor       = new Color(0.2f, 0.8f, 0.3f, 1f);
    // HALF HOLD 색상 고정 (#FF6100)
    [SerializeField] private Color halfHoldColor = new Color(1f, 97f / 255f, 0f, 1f);
    [SerializeField] private Color holdColor     = new Color(1f, 0.82f, 0.1f, 1f);
    [SerializeField] private Color failColor     = new Color(0.95f, 0.2f, 0.2f, 1f);
    [SerializeField] private bool  blinkOnFail   = true;
    [SerializeField, Range(0.1f, 5f)] private float blinkPeriod   = 1f;
    [SerializeField, Range(0f,   5f)] private float emissionBoost = 1.5f;

    [Header("Downstream (children)")]
    [Tooltip("하류 전파/참조를 위한 자식 터널들 (인스펙터에서 드래그)")]
    [SerializeField] private TunnelController[] downstreamTunnels;
    public TunnelController[] DownstreamTunnels => downstreamTunnels;

    [Header("Downstream (spawners)")]
    [Tooltip("이 터널의 HOLD/RESUME 신호를 받아 스폰을 멈출 Spawner들")]
    [SerializeField] private ProductSpawner[] downstreamSpawners;
    public ProductSpawner[] DownstreamSpawners => downstreamSpawners;

    [Header("Branch children (optional, for junction tunnels etc.)")]
    [Tooltip("갈림길 이후에 붙는 자식 터널들 (부분 정체(HALF_HOLD) 판단용)")]
    [SerializeField] private TunnelController[] branchChildren;
    // FactoryEnvManager에서 바로 쓰기 위한 프로퍼티
    public TunnelController[] BranchChildren => branchChildren;

    [Header("Hold propagation (self based)")]
    [Tooltip("큐 채움 비율 기반으로 자신/자식에게 HOLD/RESUME 전파할지 여부")]
    [SerializeField] private bool enableHoldPropagation = true;

    [Tooltip("이 비율 이하로 비면 RESUME 전파 (히스테리시스)")]
    [SerializeField, Range(0f, 1f)] private float resumeFillRatioThreshold = 0.25f;

    [Tooltip("Q+R / Capacity 가 이 비율 이상이고 자식이 일부 막혀 있으면 HALF_HOLD")]
    [SerializeField, Range(0f, 1f)] private float halfHoldRatioThreshold = 0.5f;

    [Header("External Hold (Upstream → Me)")]
    [Tooltip("상류에서 받은 HOLD/RESUME 신호를 따를지 여부")]
    [SerializeField] private bool obeyUpstreamHold = true;

    [Tooltip("외부 HOLD 참조 개수(복수 상류 지원)")]
    [SerializeField] private int upstreamHoldRefs = 0;

    [Header("Road Count (Upstream Zones)")]
    [Tooltip("터널 앞 도로 위 Product 개수를 세는 Trigger Zone들 (복수 가능)")]
    [SerializeField] private RoadProductCounter[] upstreamRoadZones;

    // ===== Debug =====
    [Header("Debug Log")]
    [SerializeField] private bool debugLogs = true;
    [SerializeField] private bool debugVerbose = false;

    // ===== Drain pacing =====
    [Header("Drain Pacing (Queue → Belt)")]
    [SerializeField] private float resumeExtraDelay = 0f;
    private float nextDrainAt = 0f;

    private bool holdSent = false;

    [SerializeField] private bool isFailed = false;
    public bool IsFailed => isFailed;

    Coroutine repairCo;
    Coroutine blinkCo;
    Material instMat;

    // Box-Muller용 캐시
    bool hasSpareNormal = false;
    float spareNormal = 0f;

    void Awake()
    {
        if (statusRenderer != null)
            instMat = statusRenderer.material;
        ApplyStatusVisual();

        // 시작 시 고장 수명 초기화
        ResetFailureCounters();
    }

    void OnDestroy()
    {
        if (repairCo != null) StopCoroutine(repairCo);
        if (blinkCo != null) StopCoroutine(blinkCo);
        if (instMat != null)
        {
#if UNITY_EDITOR
            DestroyImmediate(instMat);
#else
            Destroy(instMat);
#endif
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(debugRepairKey))
            ForceRepair();

        // 외부 HOLD가 걸려 있으면 강제 HOLD
        if (obeyUpstreamHold && upstreamHoldRefs > 0 && !IsFault && state == TunnelState.RUN)
        {
            EnterHold();
        }

        // RUN / HALF_HOLD / (HOLD 이지만 외부 HOLD는 해제된 경우) → 큐 방출 허용
        if (!IsFault && queue != null && !queue.IsEmpty &&
            (IsRun || IsHalfHold || (IsHold && upstreamHoldRefs == 0)))
        {
            if (Time.time >= nextDrainAt)
                TryDrainQueue();
        }

        // === Q(큐) + R(도로 위 product 수) 기반 HOLD/RESUME 전파 ===
        if (enableHoldPropagation && queue != null)
        {
            float fillRatio = queue.FillRatio;

            // 1) HOLD 전파 조건 (Q + R 기준)
            if (!holdSent)
            {
                int current   = queue.Count;              // Q
                int predicted = GetPredictedQueueLoad();  // Q + R
                int incoming  = Mathf.Max(0, predicted - current); // R
                bool shouldSendHold = false;

                if (debugLogs && debugVerbose)
                {
                    string tag = IsFault ? "[FAULT]" : "[QUEUE]";
                    Debug.Log(Format($"{tag} HOLD 체크 → Q={current}, R={incoming}, Total={predicted} / Capacity={queue.Capacity}"));
                }

                if (queue.Capacity > 0 && predicted >= queue.Capacity)
                {
                    shouldSendHold = true;
                }

                if (shouldSendHold)
                {
                    if (debugLogs)
                    {
                        string tag = IsFault ? "[FAULT]" : "[QUEUE]";
                        Debug.Log(Format($"{tag} HOLD 전파 → Q={current}, R={incoming}, Total={predicted} / Capacity={queue.Capacity}"));
                    }

                    // HALF_HOLD 상태에서는 자기 상태를 HOLD로 올리지 않고,
                    // 상류(자식/스포너)에게만 HOLD를 전파.
                    if (!IsHalfHold)
                    {
                        EnterHold();
                    }

                    NotifyChildrenHold();
                    holdSent = true;
                }
            }

            // 2) RESUME 전파 조건 (FillRatio 기반, 큐 기반 HOLD 해제)
            if (holdSent && fillRatio <= resumeFillRatioThreshold)
            {
                if (debugLogs)
                {
                    Debug.Log(Format($"RESUME 전파 → Fill={fillRatio:0.00}, Threshold={resumeFillRatioThreshold:0.00}"));
                }

                // 자식들에게 "큐 기반 HOLD 해제" 알리기
                NotifyChildrenResume();

                // 이 터널의 큐 기반 HOLD 플래그만 내림
                holdSent = false;

                // 브랜치가 없는 단순 라인이면 여기서 바로 RUN 복구
                if (!IsFault && upstreamHoldRefs == 0 &&
                    (branchChildren == null || branchChildren.Length == 0))
                {
                    EnterRun(withResumeDelay: false);
                }
            }
        }

        // === 갈림길 자식 상태 기반 HALF_HOLD / HOLD / RUN 판정 ===
        UpdateHalfHoldByBranchChildren();
    }

    // ===== 터널 진입 =====
    public void OnProductArrive(PathFollower follower, Transform point)
    {
        if (!follower) return;

        // 제품 개수 기반 고장 체크 (시나리오에서는 useItemFailure=false로 꺼둘 수 있음)
        if (useItemFailure && !IsFault)
        {
            itemsSinceLastFailure++;

            if (itemsSinceLastFailure >= nextFailureAtItem)
            {
                if (debugLogs)
                {
                    Debug.Log(Format($"[FAIL] itemsSinceLastFailure={itemsSinceLastFailure}, target={nextFailureAtItem}"));
                }

                SetFailed(true);   // 자동 고장은 항상 SetFailed 경로로
            }
        }

        // HOLD/FAULT/HALF_HOLD이면 큐에 적재
        if (IsFault || IsHold || IsHalfHold)
        {
            HandleArrivalWhileBuffered(follower);
            return;
        }
    }

    void HandleArrivalWhileBuffered(PathFollower follower)
    {
        if (queue == null)
        {
            follower.Pause();
            return;
        }

        if (queue.TryTakeTailSlot(out var slot))
        {
            follower.ReleaseAllReservations();
            follower.transform.SetPositionAndRotation(slot.position, slot.rotation);
            follower.enabled = false;
            queue.Enqueue(follower, slot);
        }
        else
        {
            follower.Pause();
        }
    }

    // === Reservable과의 과거 연동: 더 이상 쓰지 않지만, 컴파일 에러 방지를 위한 더미 ===
    public void OnUpstreamPathEnter(PathFollower follower)
    {
        // 레거시 호환용, 아무 것도 안 함
    }

        // ===== 큐 방출 =====
        // ===== 큐 방출 =====
    void TryDrainQueue(bool force = false)
    {
        if (queue == null || queue.IsEmpty) return;
        if (IsFault) return;

        // 방출 전 상태를 기억: '외부 HOLD는 없고, 내 로직 때문에 HOLD 중' 인지
        bool wasLocalHold = IsHold && upstreamHoldRefs == 0;

        int max  = Mathf.Max(1, dispatchPerTick);
        int sent = 0;

        while (!queue.IsEmpty && sent < max)
        {
            if (!queue.TryPopHead(out var follower, out var slot))
                break;
            if (!follower) continue;

            follower.ReleaseAllReservations();

            // 실제 위치 이동
            if (afterTunnelPoint != null)
                follower.transform.SetPositionAndRotation(afterTunnelPoint.position, afterTunnelPoint.rotation);

            // ❌ 여기서 throughput 카운트 하던 코드 제거
            // if (isSink) { totalExitedCount++; }

            follower.enabled = true;
            follower.Resume();
            sent++;
        }

        if (sent > 0)
        {
            if (wasLocalHold)
            {
                // 색만 RUN, 쿨다운은 그대로 유지
                EnterRun(withResumeDelay: false);
            }

            // HALF_HOLD일 때는 방출 템포를 느리게(쿨다운 2배)
            float effectiveCooldown = dispatchCooldown;
            if (state == TunnelState.HALF_HOLD)
            {
                effectiveCooldown *= 2f;
            }

            float cd = force ? 0f : Mathf.Max(0f, effectiveCooldown);
            nextDrainAt = Time.time + cd;
        }
    }


    // ===== 실패 / 수리 =====
    public void ForceFail() => SetFailed(true);

    /// <summary>
    /// 외부(로봇, 디버그 키 등)에서 강제로 수리할 때 사용.
    /// 자동 수리 코루틴이 돌고 있으면 멈추고, FAULT 상태를 해제한다.
    /// </summary>
    public void ForceRepair()
    {
        // 이미 고장 상태가 아니면 아무 것도 안 함
        if (!IsFault && !isFailed)
            return;

        if (repairCo != null)
        {
            StopCoroutine(repairCo);
            repairCo = null;
        }
        SetFailed(false);
    }

    void SetFailed(bool failed)
    {
        if (isFailed == failed) return;
        isFailed = failed;

        if (failed)
        {
            EnterFault();
            
            // Notify upstream spawners to slow down
            if (slowUpstreamWhenFault && upstreamSpawners != null)
            {
                foreach (var spawner in upstreamSpawners)
                {
                    if (spawner != null)
                    {
                        // Store original multiplier before changing it
                        if (!originalSpawnMultipliers.ContainsKey(spawner))
                        {
                            originalSpawnMultipliers[spawner] = spawner.GetSpawnMultiplier();
                        }
                        
                        // Apply fault speed multiplier
                        spawner.SetSpawnMultiplier(faultSpeedMultiplier, this);
                    }
                }
                
                if (debugLogs)
                {
                    Debug.Log(Format($"[SPEED] Slowing down {upstreamSpawners.Length} upstream spawners to {faultSpeedMultiplier}x speed"));
                }
            }
        }
        else
        {
            // 수리되면 제품 개수 기반 카운터/수명 리셋
            ResetFailureCounters();

            if (IsFault)
            {
                if (upstreamHoldRefs > 0)
                {
                    SetState(TunnelState.HOLD);
                }
                else
                {
                    SetState(TunnelState.RUN);
                    ArmResumeDelay();
                }
            }
            else
            {
                ApplyStatusVisual();
            }
            
            // Notify upstream spawners to return to normal speed
            if (slowUpstreamWhenFault && upstreamSpawners != null)
            {
                foreach (var spawner in upstreamSpawners)
                {
                    if (spawner != null)
                    {
                        // Restore original multiplier
                        if (originalSpawnMultipliers.ContainsKey(spawner))
                        {
                            spawner.SetSpawnMultiplier(originalSpawnMultipliers[spawner], this);
                        }
                        else
                        {
                            // Default to 1.0 if not stored
                            spawner.SetSpawnMultiplier(1.0f, this);
                        }
                    }
                }
                
                // Clear the dictionary
                originalSpawnMultipliers.Clear();
                
                if (debugLogs)
                {
                    Debug.Log(Format($"[SPEED] Restoring normal speed for {upstreamSpawners.Length} upstream spawners"));
                }
            }
        }
    }

    void ApplyStatusVisual()
    {
        if (instMat == null && statusRenderer != null)
            instMat = statusRenderer.material;
        if (instMat == null) return;

        Color targetColor = okColor;
        if (IsFault)          targetColor = failColor;
        else if (IsHold)      targetColor = holdColor;
        else if (IsHalfHold)  targetColor = halfHoldColor;

        if (instMat.HasProperty(colorProperty))
            instMat.SetColor(colorProperty, targetColor);

        if (blinkOnFail)
        {
            if (IsFault)
            {
                if (blinkCo == null)
                    blinkCo = StartCoroutine(CoBlink());
            }
            else
            {
                if (blinkCo != null)
                {
                    StopCoroutine(blinkCo);
                    blinkCo = null;
                }
                if (instMat.IsKeywordEnabled("_EMISSION"))
                    instMat.DisableKeyword("_EMISSION");
            }
        }
    }

    // ===== 상태 전환 =====
    void SetState(TunnelState newState)
    {
        if (state == newState) return;
        state = newState;
        ApplyStatusVisual();
    }

    public void EnterRun(bool withResumeDelay)
    {
        if (IsFault) return;

        SetState(TunnelState.RUN);
        if (withResumeDelay)
        {
            ArmResumeDelay();
        }
    }

    public void EnterHalfHold()
    {
        if (IsFault) return;
        SetState(TunnelState.HALF_HOLD);
    }

    public void EnterHold()
    {
        if (state == TunnelState.FAULT) return;
        SetState(TunnelState.HOLD);
    }

    public void EnterFault()
    {
        if (IsFault) return;

        SetState(TunnelState.FAULT);

        // ★ 체크되어 있을 때만 자동 수리 코루틴 동작
        if (autoRepairFixedDelay)
        {
            if (repairCo != null)
                StopCoroutine(repairCo);
            repairCo = StartCoroutine(CoAutoRepair());
        }
    }

    IEnumerator CoAutoRepair()
    {
        yield return new WaitForSeconds(fixedRepairDelay);
        SetFailed(false);
    }

    // ===== 전파 =====
    public void NotifyChildrenHold()
    {
        if (downstreamTunnels != null)
            foreach (var t in downstreamTunnels) t?.OnUpstreamHold(this);

        if (downstreamSpawners != null)
            foreach (var s in downstreamSpawners) s?.OnUpstreamHold();
    }

    public void NotifyChildrenResume()
    {
        if (downstreamTunnels != null)
            foreach (var t in downstreamTunnels) t?.OnUpstreamResume(this);

        if (downstreamSpawners != null)
            foreach (var s in downstreamSpawners) s?.OnUpstreamResume();
    }

    // 상류에서 HOLD 신호를 받았을 때
    public void OnUpstreamHold(TunnelController parent)
    {
        upstreamHoldRefs = Mathf.Max(0, upstreamHoldRefs + 1);

        if (obeyUpstreamHold)
        {
            // 상류에서 HOLD가 들어오면 나만 HOLD로 전환.
            // 내 큐(Q+R)가 꽉 차는 순간에만 Update() 내 Q+R 로직이
            // 다시 NotifyChildrenHold()를 호출해서 그 다음 상류를 HOLD로 만든다.
            // (계단식 back-pressure)
            EnterHold();
            // 여기서 더 이상 NotifyChildrenHold()를 바로 호출하지 않는다.
        }
    }

    // 상류에서 RESUME 신호를 받았을 때
    public void OnUpstreamResume(TunnelController parent)
    {
        upstreamHoldRefs = Mathf.Max(0, upstreamHoldRefs - 1);

        if (upstreamHoldRefs == 0 && !IsFault && state == TunnelState.HOLD)
        {
            holdSent = false;
            EnterRun(withResumeDelay: true);
            NotifyChildrenResume();
        }
    }

    // ===== 갈림길 HALF_HOLD 판정 =====
    void UpdateHalfHoldByBranchChildren()
    {
        if (branchChildren == null || branchChildren.Length == 0)
            return;

        int active    = 0;
        int blocked   = 0;
        int available = 0;

        foreach (var child in branchChildren)
        {
            if (child == null) continue;
            active++;

            if (child.IsHold || child.IsFault)
                blocked++;
            else
                available++;
        }

        if (active == 0) return;

        // queue가 없으면 예전 방식 그대로: 자식 state만 보고 판단
        if (queue == null || queue.Capacity <= 0)
        {
            // 자식 모두 막힘 → 나도 HOLD (완전 정지)
            if (blocked == active)
            {
                EnterHold();
                return;
            }

            // 일부만 막힘, 일부는 살아있음 → HALF_HOLD
            if (blocked > 0 && available > 0)
            {
                EnterHalfHold();
                return;
            }

            // 전부 사용 가능 상태이고, 상류 HOLD도 없고, 내가 보낸 HOLD도 아니면 RUN
            if (available == active && upstreamHoldRefs == 0 && !IsFault && !holdSent)
            {
                EnterRun(withResumeDelay: false);
            }

            return;
        }

        // 여기부터는 queue(Q+R) 압력까지 같이 고려하는 로직
        int predicted = GetPredictedQueueLoad(); // Q + R
        float ratio   = queue.Capacity > 0 ? (float)predicted / queue.Capacity : 0f;

        // 자식 모두 막히면 ratio 상관없이 무조건 HOLD
        if (blocked == active)
        {
            EnterHold();
            return;
        }

        // ⚙️ 일부만 막힘, 일부는 살아있음 → 항상 HALF_HOLD (ratio와 무관)
        if (blocked > 0 && available > 0)
        {
            EnterHalfHold();
            return;
        }

        // 자식 다 살아있음 + 상류 HOLD 없고 + 내가 큐 기반 HOLD 안 보낸 상태 → RUN 복귀
        if (available == active && upstreamHoldRefs == 0 && !IsFault && !holdSent)
        {
            EnterRun(withResumeDelay: false);
        }
    }

    IEnumerator CoBlink()
    {
        if (instMat == null) yield break;

        instMat.EnableKeyword("_EMISSION");
        float t = 0f;

        while (state == TunnelState.FAULT)
        {
            t += Time.deltaTime;
            float s = 0.5f + 0.5f * Mathf.Sin((t / Mathf.Max(0.01f, blinkPeriod)) * Mathf.PI * 2f);
            if (instMat.HasProperty("_EmissionColor"))
                instMat.SetColor("_EmissionColor", failColor * (emissionBoost * s));
            yield return null;
        }

        if (instMat.IsKeywordEnabled("_EMISSION"))
            instMat.DisableKeyword("_EMISSION");
    }

#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        Color c = okColor;
        if (IsFault)         c = failColor;
        else if (IsHold)     c = holdColor;
        else if (IsHalfHold) c = halfHoldColor;

        Gizmos.color = c;
        Gizmos.DrawWireCube(transform.position, new Vector3(1, 1, 1));

        if (afterTunnelPoint != null)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(transform.position, afterTunnelPoint.position);
        }

            // Draw lines to upstream spawners
        if (upstreamSpawners != null)
        {
            Gizmos.color = Color.magenta;
            foreach (var spawner in upstreamSpawners)
            {
                if (spawner != null)
                {
                    Gizmos.DrawLine(transform.position, spawner.transform.position);
                    // Draw an arrow
                    Vector3 direction = (spawner.transform.position - transform.position).normalized;
                    Vector3 arrowTip = transform.position + direction * 1.5f;
                    Gizmos.DrawLine(arrowTip, arrowTip - Quaternion.Euler(0, 45, 0) * direction * 0.3f);
                    Gizmos.DrawLine(arrowTip, arrowTip - Quaternion.Euler(0, -45, 0) * direction * 0.3f);
                }
            }
        }
    }
#endif

    // 현재 큐 개수 + "터널 앞 도로 존들 안에 있는 애들(R)" 합산
    int GetPredictedQueueLoad()
    {
        if (queue == null)
            return 0;

        int current = queue.Count;  // Q

        int incoming = 0;           // R
        if (upstreamRoadZones != null)
        {
            for (int i = 0; i < upstreamRoadZones.Length; i++)
            {
                var zone = upstreamRoadZones[i];
                if (zone != null)
                    incoming += zone.Count;
            }
        }

        int total = current + incoming; // Q + R

        if (debugLogs && debugVerbose)
        {
            Debug.Log(Format($"[PREDICT] Q={current}, R={incoming}, Total={total} / Capacity={queue.Capacity}"));
        }

        return total;
    }

    // ===== Failure Items 수명 리셋 & 샘플링 =====
    // ===== Failure Items 수명 리셋 & 샘플링 =====
    void ResetFailureCounters()
    {
        itemsSinceLastFailure = 0;

        if (useItemFailure)
        {
            nextFailureAtItem = SampleNextFailureItem();
        }
        else
        {
            // 시나리오 모드 등에서 확률 고장 끌 때는 사실상 고장 안 나게 막아둠
            nextFailureAtItem = int.MaxValue;
        }

        if (debugLogs)
        {
            Debug.Log(Format($"[FAILURE RESET] nextFailureAtItem={nextFailureAtItem} (model={failureModel}, useItemFailure={useItemFailure})"));
        }
    }

    int SampleNextFailureItem()
    {
        switch (failureModel)
        {
            case FailureModel.Constant:
                return SampleConstantItems();

            case FailureModel.Gaussian:
                return SampleGaussianItems();

            case FailureModel.Exponential:
                return SampleExponentialItems();
        }

        return int.MaxValue; // 사실상 고장 안 남
    }

    int SampleConstantItems()
    {
        // 평균 M개당 한 번 고장 → 지수/기하 분포로 근사
        if (constantMeanItemsToFailure <= 0f)
            return int.MaxValue;

        float M = Mathf.Max(1f, constantMeanItemsToFailure);
        float u = Mathf.Clamp01(URandom.value);
        u = Mathf.Max(u, 1e-6f);

        // 지수분포: N ~ Exp(1/M)
        float x = -Mathf.Log(u) * M;
        int n = Mathf.Max(1, Mathf.RoundToInt(x));
        return n;
    }

    int SampleGaussianItems()
    {
        if (gaussianStdItems <= 0f)
            return Mathf.Max(gaussianMinItems, 1);

        float mu  = Mathf.Max(1f, gaussianMeanItems);
        float sig = Mathf.Max(1f, gaussianStdItems);

        float g = NextStandardNormal(); // 평균 0, 분산 1
        float x = mu + sig * g;

        int n = Mathf.Max(gaussianMinItems, Mathf.RoundToInt(x));
        return n;
    }

    int SampleExponentialItems()
    {
        if (exponentialMeanItems <= 0f)
            return int.MaxValue;

        float M = Mathf.Max(1f, exponentialMeanItems);
        float u = Mathf.Clamp01(URandom.value);
        u = Mathf.Max(u, 1e-6f);

        float x = -Mathf.Log(u) * M; // Exp(1/M)
        int n = Mathf.Max(exponentialMinItems, Mathf.RoundToInt(x));
        return n;
    }

    // 표준 정규분포 샘플 (Box-Muller)
    float NextStandardNormal()
    {
        if (hasSpareNormal)
        {
            hasSpareNormal = false;
            return spareNormal;
        }

        float u1 = Mathf.Clamp01(URandom.value);
        float u2 = Mathf.Clamp01(URandom.value);

        u1 = Mathf.Max(u1, 1e-6f);

        float r = Mathf.Sqrt(-2f * Mathf.Log(u1));
        float theta = 2f * Mathf.PI * u2;

        float z0 = r * Mathf.Cos(theta);
        float z1 = r * Mathf.Sin(theta);

        spareNormal = z1;
        hasSpareNormal = true;

        return z0;
    }

    // ===== Helpers =====
    void ArmResumeDelay()
    {
        float extra = Mathf.Max(0f, resumeExtraDelay);
        nextDrainAt = Time.time + extra;
    }

    string Format(string msg) => $"[TunnelController:{name}] {msg}";
}
