using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class QueueZone : MonoBehaviour
{
    [Header("Slots (모두 슬롯입니다 — 엔트리 없음)")]
    public List<Transform> slots = new List<Transform>();

    [Tooltip("true면 리스트의 '마지막 요소'가 꼬리(먼저 채움). false면 '첫 요소'가 꼬리.")]
    public bool tailIsLastInList = true;

    // 점유 현황
    private readonly HashSet<Transform> _occ = new HashSet<Transform>();

    // 슬롯 <-> 대기 제품 매핑 + FIFO 순서(머리=먼저 나갈 제품)
    private readonly Dictionary<Transform, PathFollower> _holder = new Dictionary<Transform, PathFollower>();
    private readonly LinkedList<Transform> _order = new LinkedList<Transform>();

    /// <summary>총 슬롯 개수.</summary>
    public int Capacity => (slots != null ? slots.Count : 0);

    /// <summary>현재 점유 중인 슬롯 수.</summary>
    public int OccupiedCount => _occ.Count;

    /// <summary>큐 안에 있는 제품 수 (OccupiedCount와 동일, alias).</summary>
    public int Count => OccupiedCount;

    /// <summary>채움 비율(0.0~1.0). Capacity==0이면 0 반환.</summary>
    public float FillRatio => Capacity <= 0 ? 0f : (float)OccupiedCount / Capacity;

    /// <summary>꼬리에서부터 비어있는 슬롯 하나를 점유해서 반환.</summary>
    public bool TryTakeTailSlot(out Transform slot)
    {
        if (slots == null || slots.Count == 0)
        {
            slot = null; 
            return false;
        }

        if (tailIsLastInList)
        {
            for (int i = slots.Count - 1; i >= 0; --i)
            {
                var s = slots[i];
                if (!s || _occ.Contains(s)) continue;
                _occ.Add(s);
                slot = s;
                return true;
            }
        }
        else
        {
            for (int i = 0; i < slots.Count; ++i)
            {
                var s = slots[i];
                if (!s || _occ.Contains(s)) continue;
                _occ.Add(s);
                slot = s;
                return true;
            }
        }

        slot = null;
        return false;
    }

    /// <summary>슬롯에 텔레포트된 제품을 큐(FIFO)에 등록.</summary>
    public void Enqueue(PathFollower follower, Transform slot)
    {
        if (!slot || follower == null) return;
        _holder[slot] = follower;
        _order.AddLast(slot);
    }

    /// <summary>머리(먼저 들어온 순)에서 하나 꺼냄.</summary>
    public bool TryPopHead(out PathFollower follower, out Transform slot)
    {
        if (_order.Count == 0)
        {
            follower = null; 
            slot = null; 
            return false;
        }

        slot = _order.First.Value;
        _order.RemoveFirst();

        if (!_holder.TryGetValue(slot, out follower))
            follower = null;
        _holder.Remove(slot);

        _occ.Remove(slot);
        return follower != null;
    }

    /// <summary>슬롯 점유 해제만 필요할 때.</summary>
    public void FreeSlot(Transform slot)
    {
        if (!slot) return;
        _occ.Remove(slot);
        _holder.Remove(slot);
        _order.Remove(slot);
    }

    public bool IsFull  => slots != null && _occ.Count >= slots.Count;
    public bool IsEmpty => _order.Count == 0;
}
