using UnityEngine;

/// 전달받은 Transform들의 "위치"를 복제해서
/// 일회성 TargetPath 오브젝트를 만들어 준다.
public static class RuntimePathFactory
{
    public static TargetPath Create(string name, Transform[] waypoints)
    {
        var root = new GameObject(name);
        var tp = root.AddComponent<TargetPath>();

        if (waypoints == null || waypoints.Length < 2) return tp;

        for (int i = 0; i < waypoints.Length; i++)
        {
            var src = waypoints[i];
            var c = new GameObject($"pt{i}");
            c.transform.SetParent(root.transform);
            c.transform.position = src.position;
            c.transform.rotation = Quaternion.identity;
        }

        return tp;
    }
}
