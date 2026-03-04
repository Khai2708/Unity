using UnityEngine;
using System.Text.RegularExpressions;

public enum PathOrderMode { SiblingOrder, NameNumberAsc, NameNumberDesc }

[DisallowMultipleComponent]
public class TargetPath : MonoBehaviour
{
    [Header("Order of child points")]
    public PathOrderMode orderMode = PathOrderMode.NameNumberAsc;

    [SerializeField, HideInInspector] private Transform[] points;
    public int Count => points?.Length ?? 0;

    public Transform GetPoint(int i)
    {
        if (Count == 0) return null;
        i = Mathf.Clamp(i, 0, Count - 1);
        return points[i];
    }

#if UNITY_EDITOR
    void Reset()      => Cache();
    void OnValidate() => Cache();

    static readonly Regex tailNum = new Regex(@".*?\((\d+)\)\s*$", RegexOptions.Compiled);

    int KeyOf(Transform t)
    {
        // 이름 끝의 숫자 "(123)" 추출, 없으면 형제 인덱스로 대체
        var m = tailNum.Match(t.name);
        if (m.Success && int.TryParse(m.Groups[1].Value, out int n))
            return n;
        return t.GetSiblingIndex(); // 숫자 없으면 sibling order
    }

    void Cache()
    {
        int n = transform.childCount;
        points = new Transform[n];
        for (int i = 0; i < n; i++) points[i] = transform.GetChild(i);

        System.Array.Sort(points, (a, b) =>
        {
            switch (orderMode)
            {
                case PathOrderMode.SiblingOrder:
                    return a.GetSiblingIndex().CompareTo(b.GetSiblingIndex());
                case PathOrderMode.NameNumberAsc:
                    return KeyOf(a).CompareTo(KeyOf(b));
                case PathOrderMode.NameNumberDesc:
                    return KeyOf(b).CompareTo(KeyOf(a));
            }
            return 0;
        });
    }

    void OnDrawGizmos()
    {
        if (Count == 0) return;
        Gizmos.color = new Color(1f, 0.6f, 0.1f, 1f);
        for (int i = 0; i < Count; i++)
        {
            var p = points[i] ? points[i].position : transform.position;
            Gizmos.DrawSphere(p, 0.08f);
            if (i < Count - 1 && points[i+1])
                Gizmos.DrawLine(p, points[i+1].position);
        }
    }
#endif
}
