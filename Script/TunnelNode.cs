using UnityEngine;

[DisallowMultipleComponent]
public class TunnelNode : MonoBehaviour
{
    [Header("Wiring")]
    [Tooltip("이 터널의 동작을 관리하는 컨트롤러")]
    public TunnelController controller;

    [Tooltip("터널 앞 대기 위치(선택). 비워두면 현재 노드 위치 사용")]
    public Transform holdBeforeTunnel;

    /// <summary>
    /// 호환용 오버로드. 호출부가 포인트 Transform을 넘기지 않는 경우를 위해
    /// 현재 노드의 transform을 전달해 OnArrive(PathFollower, Transform)을 호출합니다.
    /// </summary>
    public void OnArrive(PathFollower follower)
    {
        OnArrive(follower, transform);
    }

    /// <summary>
    /// 터널 노드(RoadPath가 Tunnel 타입인 지점)에 제품이 도착했을 때 호출됩니다.
    /// TunnelController로 제어를 위임합니다.
    /// </summary>
    /// <param name="follower">이동 중인 제품(팔로워)</param>
    /// <param name="point">도착한 Waypoint의 Transform</param>
    public void OnArrive(PathFollower follower, Transform point)
    {
        if (follower == null) return;

        // 컨트롤러가 누락되었으면 부모/자식에서 한 번 찾아보고 그래도 없으면 종료
        if (controller == null)
        {
            controller = GetComponentInParent<TunnelController>();
            if (controller == null)
                controller = GetComponent<TunnelController>();

            if (controller == null)
            {
                Debug.LogWarning($"[TunnelNode] TunnelController reference is missing on {name}.", this);
                return;
            }
        }

        controller.OnProductArrive(follower, point);
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        // 노드 위치 표시
        Gizmos.color = new Color(0.0f, 0.8f, 1.0f, 0.9f);
        Gizmos.DrawWireSphere(transform.position, 0.12f);

        // 컨트롤러 방향 라인
        if (controller != null)
        {
            Gizmos.DrawLine(transform.position, controller.transform.position);
        }

        // 대기 포인트 표시
        if (holdBeforeTunnel != null)
        {
            Gizmos.color = new Color(1.0f, 0.85f, 0.2f, 0.9f);
            Gizmos.DrawWireSphere(holdBeforeTunnel.position, 0.1f);
            Gizmos.DrawLine(transform.position, holdBeforeTunnel.position);
        }
    }
#endif
}
