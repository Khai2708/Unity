using UnityEngine;

/// <summary>
/// PathFollower가 웨이포인트에 도달했을 때,
/// 해당 포인트가 Tunnel 타입이면 TunnelNode → TunnelController로
/// 도착 이벤트를 라우팅해준다.
/// 제품(박스) 프리팹에 이 컴포넌트를 붙여주세요.
/// </summary>
[RequireComponent(typeof(PathFollower))]
public class WaypointGateRouter : MonoBehaviour
{
    private PathFollower follower;

    private void Awake()
    {
        follower = GetComponent<PathFollower>();
        // PathFollower 내부에서 웨이포인트 도달 시 이 이벤트를 호출한다고 가정
        follower.OnReachedPoint += HandleReached;
    }

    private void OnDestroy()
    {
        if (follower != null) follower.OnReachedPoint -= HandleReached;
    }

    private void HandleReached(Transform point, WaypointMarker mk)
    {
        if (point == null || mk == null) return;
        if (mk.type != WaypointType.Tunnel) return;

        var node = point.GetComponent<TunnelNode>();
        if (node != null)
        {
            node.OnArrive(follower); // → TunnelController.HandleArrivalAtTunnel(...)까지 전달
        }
    }
}
