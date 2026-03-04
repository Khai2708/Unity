using UnityEngine;
using System.Collections.Generic;

[DisallowMultipleComponent]
public class Reservable : MonoBehaviour
{
    [SerializeField]
    private PathFollower holder;   // 이 포인트를 점유 중인 PathFollower

    [Header("Auto-release safety")]
    [Tooltip("holder가 너무 멀리 떨어지면 자동으로 락을 해제할지 여부\n" +
             "→ 큐 앞 버퍼 센서로 쓰는 포인트는 보통 끄는 걸 추천")]
    public bool useAutoRelease = false;   // 기본값: 끔

    [Tooltip("holder와 이 포인트 사이 거리가 이 값보다 커지면 자동 해제")]
    public float autoReleaseDistance = 0.5f;

    [Header("Upstream Tunnels (optional)")]
    [Tooltip("이 Reservable을 '시작점'으로 사용하는 TunnelController들\n" +
             "→ 처음 점유될 때 해당 터널에게 Path 진입을 알려준다.")]
    [SerializeField]
    private List<TunnelController> upstreamTunnels = new List<TunnelController>();

    [Header("Debug")]
    [Tooltip("TryReserve / Release / AutoRelease 호출을 전부 콘솔에 찍을지 여부")]
    public bool verboseLog = false;

    public PathFollower Holder => holder;
    public bool IsFree => holder == null;
    public bool IsHeldBy(PathFollower who) => holder != null && holder == who;

    /// <summary>
    /// TunnelController가 Awake에서 나를 등록할 때 호출.
    /// (같은 터널이 여러 번 등록하려고 해도 한 번만 들어가게 처리)
    /// </summary>
    public void RegisterUpstreamTunnel(TunnelController t)
    {
        if (t == null) return;
        if (!upstreamTunnels.Contains(t))
            upstreamTunnels.Add(t);
    }

    /// <summary>
    /// 이 포인트를 점유 시도. 이미 내가 잡고 있으면 true, 비어 있으면 내가 잡고 true, 아니면 false.
    /// </summary>
    public bool TryReserve(PathFollower who)
    {
        string whoName    = (who    != null) ? who.name    : "null";
        string holderName = (holder != null) ? holder.name : "null";

        if (verboseLog)
            Debug.Log($"[Reservable:{name}] TryReserve by {whoName} (current holder={holderName})");

        if (who == null)
            return false;

        // 이미 내가 잡고 있으면 OK
        if (holder == who)
        {
            if (verboseLog)
                Debug.Log($"[Reservable:{name}] → already held by same, OK");
            return true;
        }

        // 비어 있으면 내가 잡는다
        if (holder == null)
        {
            holder = who;

            if (verboseLog)
                Debug.Log($"[Reservable:{name}] → acquired by {whoName} (was free)");

            // ★ 이 Reservable을 시작점으로 사용하는 터널들에게
            // "이 follower가 너네 path 타기 시작했다"라고 알려준다.
            if (upstreamTunnels != null && upstreamTunnels.Count > 0)
            {
                foreach (var t in upstreamTunnels)
                {
                    t?.OnUpstreamPathEnter(who);
                }
            }

            return true;
        }

        // 다른 애가 잡고 있음
        if (verboseLog)
            Debug.Log($"[Reservable:{name}] → FAILED (held by {holderName})");
        return false;
    }

    /// <summary>
    /// 주어진 PathFollower가 나갈 때만 락 해제
    /// </summary>
    public void Release(PathFollower who)
    {
        string whoName    = (who    != null) ? who.name    : "null";
        string holderName = (holder != null) ? holder.name : "null";

        if (verboseLog)
            Debug.Log($"[Reservable:{name}] Release by {whoName} (current holder={holderName})");

        if (who != null && holder == who)
        {
            if (verboseLog)
                Debug.Log($"[Reservable:{name}] → holder cleared");
            holder = null;
        }
    }

    /// <summary>
    /// 강제 해제 (디버그용 / 특수 상황)
    /// </summary>
    public void ForceRelease()
    {
        if (verboseLog)
        {
            string holderName = (holder != null) ? holder.name : "null";
            Debug.Log($"[Reservable:{name}] ForceRelease (holder={holderName} → null)");
        }
        holder = null;
    }

    private void Update()
    {
        // 세이프티: holder가 너무 멀리 떨어지면 자동으로 락 해제
        if (!useAutoRelease || holder == null)
            return;

        float dist = Vector3.Distance(transform.position, holder.transform.position);

        if (dist > autoReleaseDistance)
        {
            if (verboseLog)
            {
                string holderName = (holder != null) ? holder.name : "null";
                Debug.Log($"[Reservable:{name}] AutoRelease: holder={holderName}, dist={dist:0.000} > {autoReleaseDistance}");
            }
            holder = null;
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        // holder가 잡혀 있으면 노란색으로 표시 (버퍼 포인트 시각화용)
        if (holder != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, 0.05f);
        }
    }
#endif
}
