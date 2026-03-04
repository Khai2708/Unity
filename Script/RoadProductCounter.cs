using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Collider))]
public class RoadProductCounter : MonoBehaviour
{
    // 이 존 안에 "실제로 존재하는" PathFollower 들
    private HashSet<PathFollower> inside = new HashSet<PathFollower>();

    /// <summary>
    /// 현재 존 안에 있는 product 개수
    /// </summary>
    public int Count => inside.Count;

    void OnTriggerEnter(Collider other)
    {
        // Product 오브젝트 구조에 따라 바로 GetComponent<PathFollower>()거나
        // 상위에서 가져와야 할 수 있음. (지금은 부모까지 찾는 버전)
        var pf = other.GetComponentInParent<PathFollower>();
        if (pf == null) return;

        inside.Add(pf);
    }

    void OnTriggerExit(Collider other)
    {
        var pf = other.GetComponentInParent<PathFollower>();
        if (pf == null) return;

        inside.Remove(pf);
    }

    // 필요하면 디버그용으로 목록 보고싶을 때
    public IReadOnlyCollection<PathFollower> CurrentProducts => inside;
}
