using UnityEngine;

[DisallowMultipleComponent]
public class JunctionPoint : MonoBehaviour
{
    [Header("Upstream tunnel (이 분기 직전 터널, 예: Tunnel_1)")]
    [Tooltip("이 Junction을 기준으로 half-hold 여부를 판단할 부모 터널")]
    public TunnelController parentTunnel;

    [System.Serializable]
    public class Branch
    {
        [Tooltip("이 브랜치 식별용 이름 (디버그용)")]
        public string name;

        [Tooltip("이 브랜치로 갈 때 탈 TargetPath (브릿지 경로)")]
        public TargetPath targetPath;

        [Tooltip("targetPath 안에서 시작할 인덱스 (보통 0)")]
        public int startIndex = 0;

        [Header("Downstream tunnel (선택)")]
        [Tooltip("이 브랜치 뒤에 이어지는 TunnelController (예: Tunnel_2-1, Tunnel_2-2)")]
        public TunnelController downstreamTunnel;

        [Header("Base probability")]
        [Range(0f, 1f)]
        [Tooltip("부모 터널이 half-hold가 아닐 때 사용하는 기본 확률 가중치")]
        public float baseProbability = 1f;
    }

    [Header("Branches from this junction")]
    [Tooltip("갈림길별 브랜치 설정 (최소 1~2개)")]
    public Branch[] branches;


    /// <summary>
    /// PathFollower가 이 포인트에 도달했을 때 PathFollower.ReachPoint()에서 호출됨
    /// </summary>
    public void TryRedirect(PathFollower follower, TargetPath currentPath, int pointIndex)
    {
        if (follower == null) return;
        if (branches == null || branches.Length == 0) return;

        bool parentIsHalfHold = (parentTunnel != null && parentTunnel.IsHalfHold);

        // 1) 후보 브랜치 목록 만들기
        //    - half-hold면: downstreamTunnel이 HOLD/FAULT인 브랜치는 제외
        //    - 아니면: targetPath만 있으면 전부 후보
        System.Collections.Generic.List<Branch> candidates =
            new System.Collections.Generic.List<Branch>();

        if (parentIsHalfHold)
        {
            foreach (var b in branches)
            {
                if (b == null || b.targetPath == null)
                    continue;

                bool blocked = false;
                if (b.downstreamTunnel != null)
                {
                    if (b.downstreamTunnel.IsHold || b.downstreamTunnel.IsFault)
                        blocked = true;
                }

                if (!blocked)
                    candidates.Add(b);
            }
        }
        else
        {
            foreach (var b in branches)
            {
                if (b != null && b.targetPath != null)
                    candidates.Add(b);
            }
        }

        // 2) 적합한 브랜치가 하나도 없으면 → 아무 것도 하지 않고 기존 path 유지
        if (candidates.Count == 0)
        {
            // 예: 부모가 half-hold이고, 등록된 두 브랜치 터널이 모두 HOLD/FAULT인 경우
            // 그냥 현재 타고 있는 path 그대로 진행 (또는 나중에 Pause/Queue 등으로 확장 가능)
            return;
        }

        // 3) 후보가 1개면 그냥 그걸 선택
        Branch chosen = null;
        if (candidates.Count == 1)
        {
            chosen = candidates[0];
        }
        else
        {
            // 4) 후보가 여러 개면 baseProbability 기반 가중 랜덤
            float totalW = 0f;
            foreach (var b in candidates)
                totalW += Mathf.Max(0f, b.baseProbability);

            if (totalW <= 0f)
            {
                // 전부 0이면 균등 랜덤
                int idx = Mathf.FloorToInt(Random.value * candidates.Count);
                if (idx >= candidates.Count) idx = candidates.Count - 1;
                chosen = candidates[idx];
            }
            else
            {
                float r = Random.value * totalW;
                float acc = 0f;
                foreach (var b in candidates)
                {
                    float w = Mathf.Max(0f, b.baseProbability);
                    acc += w;
                    if (r <= acc)
                    {
                        chosen = b;
                        break;
                    }
                }

                // 혹시 못 뽑힌 경우 대비
                if (chosen == null)
                    chosen = candidates[candidates.Count - 1];
            }
        }

        if (chosen == null || chosen.targetPath == null)
            return;

        int idxStart = Mathf.Max(0, chosen.startIndex);
        follower.SwitchPath(chosen.targetPath, idxStart, true);
    }
}
