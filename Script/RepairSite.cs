using UnityEngine;

[DisallowMultipleComponent]
public class RepairSite : MonoBehaviour
{
    [Header("Wiring")]
    [Tooltip("이 수리 사이트가 담당하는 터널")]
    public TunnelController tunnel;

    [Tooltip("터널 앞 버퍼(QueueZone) - 옵션")]
    public QueueZone buffer;

    [Tooltip("로봇이 도착해서 설 위치. 비워두면 이 오브젝트 위치 사용")]
    public Transform repairPoint;

    [Header("고장 조건 옵션")]
    [Tooltip("버퍼가 가득 찬 경우도 '고장으로 간주'할지 여부")]
    public bool useBufferFullAsFault = false;

    [Header("Repair Queue 상태 (내부용)")]
    [HideInInspector]
    public bool isQueued = false;

    [Header("Repair Visual (게이지)")]
    [Tooltip("수리 중일 때 표시할 게이지 오브젝트 (예: 큐브, 캔버스 등)")]
    public Transform repairGauge;

    [Tooltip("게이지 시작 스케일 (0% 일 때)")]
    public Vector3 gaugeStartScale = new Vector3(1f, 0f, 1f);

    [Tooltip("게이지 풀 스케일 (100% 일 때)")]
    public Vector3 gaugeFullScale = new Vector3(1f, 1f, 1f);

    [Tooltip("수리 중일 때만 게이지를 활성화할지 여부")]
    public bool hideGaugeWhenIdle = true;

    float currentProgress = 0f;

    // 외부에서 로봇이 쓰는 수리 포인트
    public Transform RepairPoint => repairPoint != null ? repairPoint : transform;

    /// <summary>
    /// 지금 이 기계가 "수리 대상"인지 여부.
    /// 현재 기본 로직: 실제 FAULT 상태만 true.
    /// </summary>
    public bool NeedsRepair
    {
        get
        {
            if (tunnel == null) return false;

            // 1) 실제 터널 고장일 때
            if (tunnel.IsFault)
                return true;

            // 2) 옵션: 버퍼가 꽉 찬 경우까지 고장으로 보고 싶으면
            if (useBufferFullAsFault && buffer != null)
            {
                // QueueZone에 IsFull 같은 속성이 있으면 사용
                // 예: if (buffer.IsFull) return true;
            }

            return false;
        }
    }

    /// <summary>
    /// 수리 시작 시 (코루틴 시작 직전에) 호출
    /// </summary>
    public void BeginRepairVisual()
    {
        currentProgress = 0f;
        if (repairGauge != null)
        {
            if (hideGaugeWhenIdle)
                repairGauge.gameObject.SetActive(true);

            repairGauge.localScale = gaugeStartScale;
        }
    }

    /// <summary>
    /// 수리 진행률 0~1 업데이트
    /// </summary>
    public void UpdateRepairVisual(float progress01)
    {
        currentProgress = Mathf.Clamp01(progress01);
        if (repairGauge != null)
        {
            repairGauge.localScale = Vector3.Lerp(gaugeStartScale, gaugeFullScale, currentProgress);
        }
    }
        void Start()
    {
        // 시작할 때 기본 게이지 상태 정리
        if (repairGauge != null)
        {
            // 스케일도 시작 스케일로 맞춰주고
            repairGauge.localScale = gaugeStartScale;

            // hideGaugeWhenIdle 이면 처음엔 꺼둔다
            if (hideGaugeWhenIdle)
                repairGauge.gameObject.SetActive(false);
            else
                repairGauge.gameObject.SetActive(true);
        }
    }

    /// <summary>
    /// 수리 완료 시(코루틴 끝) 호출
    /// </summary>
    public void EndRepairVisual()
    {
        if (repairGauge != null && hideGaugeWhenIdle)
        {
            repairGauge.gameObject.SetActive(false);
        }
    }

    /// <summary>
    /// 로봇이 이 수리 포인트에서 수리를 끝냈을 때 호출.
    /// </summary>
    public void OnRepaired()
    {
        // 다음 고장 때 다시 큐에 들어갈 수 있도록 플래그 초기화
        isQueued = false;

        if (tunnel != null)
        {
            // TunnelController에서 고장 플래그 및 상태 복구
            tunnel.ForceRepair();
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (RepairPoint != null)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(RepairPoint.position, 0.25f);
        }
    }
#endif
}
