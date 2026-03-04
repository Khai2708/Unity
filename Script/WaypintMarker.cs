using UnityEngine;

public enum WaypointType
{
    Road = 0,
    Tunnel,
    QueueEntry,
    QueueSlot,
    QueueExit,           // (미사용 가능)
    HoldBeforeTunnel     // 큐 Full 시 정지 위치
}

public class WaypointMarker : MonoBehaviour
{
    public WaypointType type = WaypointType.Road;

#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        Color c = type switch {
            WaypointType.Tunnel         => new Color(1f, 0.25f, 0.25f, 1f),
            WaypointType.QueueEntry     => new Color(1f, 0.7f,  0.2f, 1f),
            WaypointType.QueueSlot      => new Color(0.25f, 0.9f, 0.35f, 1f),
            WaypointType.QueueExit      => new Color(0.45f, 0.6f,  1f, 1f),
            WaypointType.HoldBeforeTunnel=>new Color(0.9f,  0.9f,  0.2f, 1f),
            _                           => new Color(1f, 0.6f,  0.1f, 1f)
        };
        Gizmos.color = c;
        Gizmos.DrawSphere(transform.position, 0.09f);
    }
#endif
}
