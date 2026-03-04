using UnityEngine;

public class WaypointGizmo : MonoBehaviour
{
    public Color color = new Color(1f, 0.6f, 0f, 1f);
    public float radius = 0.08f;
    public bool loop = false; // 마지막→처음 선 연결 여부

    void OnDrawGizmos()
    {
        int n = transform.childCount;
        if (n == 0) return;

        Gizmos.color = color;

        // 점
        for (int i = 0; i < n; i++)
            Gizmos.DrawSphere(transform.GetChild(i).position, radius);

        // 선
        for (int i = 0; i < n - 1; i++)
        {
            var a = transform.GetChild(i).position;
            var b = transform.GetChild(i + 1).position;
            Gizmos.DrawLine(a, b);
        }
        if (loop && n > 1)
            Gizmos.DrawLine(transform.GetChild(n - 1).position, transform.GetChild(0).position);
    }
}
