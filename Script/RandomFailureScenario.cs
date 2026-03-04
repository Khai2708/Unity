using System.Collections.Generic;
using UnityEngine;

/// <summary>
///  - 터널 고장 시나리오를 A/B/C 3종류로 운용한다.
///    A: 3개 동시 고장, B: 2개 동시 고장, C: 1개 고장.
///  - 한 "사이클" 안에서 A/B/C가 각각 cycleCountA / B / C 번씩 등장하도록 함.
///    (예: 5,5,5 로 두면 3*5 + 2*5 + 1*5 = 30 step)
///  - 각 시나리오 타입의 순서는 랜덤이지만, 개수는 정확히 맞춘다.
///  - 한 시나리오에서 발생한 FAULT가 모두 수리되면 delayBetweenScenarios 후 다음 시나리오 시작.
/// </summary>

// ===== MOVE NodeToggle class OUTSIDE the main class =====
[System.Serializable]
public class NodeToggle
{
    public int nodeId;
    public bool isEnabled = true;
    public string tunnelName; // For display purposes
}

public class RandomFailureScenario : MonoBehaviour
{
    [Header("References")]
    public FactoryEnvManager factoryEnv;   // 인스펙터에서 할당 (없으면 자동 탐색)

    [Header("Scenario Settings")]
    [Tooltip("A: 3개 고장 시나리오가 한 사이클에서 몇 번 등장할지")]
    public int cycleCountA = 5;   // A: 3 faults
    [Tooltip("B: 2개 고장 시나리오가 한 사이클에서 몇 번 등장할지")]
    public int cycleCountB = 5;   // B: 2 faults
    [Tooltip("C: 1개 고장 시나리오가 한 사이클에서 몇 번 등장할지")]
    public int cycleCountC = 5;   // C: 1 fault

    [Tooltip("모든 고장이 수리된 후 다음 시나리오까지 대기 시간 (초)")]
    public float delayBetweenScenarios = 10f;

    [Header("Node Selection")]
    [Tooltip("Enable to select specific nodes. Disable to use all nodes.")]
    public bool useNodeFilter = false;
    
    [Tooltip("List of nodes with toggle status")]
    public List<NodeToggle> nodeToggles = new List<NodeToggle>(); // <-- This is line ~34

    [Header("Debug")]
    public bool debugLogs = true;

    // 내부 상태
    private List<TunnelController> allTunnels = new List<TunnelController>();
    private List<TunnelController> currentFaults = new List<TunnelController>();
    private bool scenarioActive = false;
    private float nextScenarioStartTime = 0f;

    // 사이클(에피소드 비슷한 단위) 내에서 남은 A/B/C 개수
    private int remainingA;
    private int remainingB;
    private int remainingC;

    void Start()
    {
        if (factoryEnv == null)
        {
            factoryEnv = FactoryEnvManager.Instance;
        }

        if (factoryEnv == null)
        {
            Debug.LogError("[RandomFailureScenario] FactoryEnvManager를 찾을 수 없습니다.");
            enabled = false;
            return;
        }

        // 공장에 등록된 터널 리스트 가져오기 + 이 시나리오 동안은 확률 고장 끄기
        if (factoryEnv.tunnels != null)
        {
            foreach (var t in factoryEnv.tunnels)
            {
                if (t != null && t.nodeId >= 0)
                {
                    // 시나리오 모드에서는 기존 items-based 확률 고장을 잠깐 꺼둔다
                    t.useItemFailure = false;
                    allTunnels.Add(t);
                }
            }
        }

        if (allTunnels.Count == 0)
        {
            Debug.LogError("[RandomFailureScenario] 사용할 터널이 없습니다.");
            enabled = false;
            return;
        }

        // Initialize nodeToggles if empty
        if (nodeToggles.Count == 0)
        {
            foreach (var tunnel in allTunnels)
            {
                if (tunnel != null)
                {
                    nodeToggles.Add(new NodeToggle { 
                        nodeId = tunnel.nodeId, 
                        isEnabled = true,
                        tunnelName = tunnel.name
                    });
                }
            }
            
            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] nodeToggles 초기화: {nodeToggles.Count}개 노드 추가됨");
            }
        }

        // 첫 사이클 초기화
        ResetCycleCounts();

        // 게임 시작하자마자 첫 시나리오 바로 시작
        StartNewScenario();
    }

    void Update()
    {
        if (!scenarioActive)
        {
            // 시나리오 사이 대기 중
            if (Time.time >= nextScenarioStartTime)
            {
                StartNewScenario();
            }
            return;
        }

        // 시나리오 진행 중이면 "모든 고장이 수리됐는지" 확인
        bool allRepaired = true;

        foreach (var t in currentFaults)
        {
            if (t == null) continue;

            // 아직 FAULT 상태인 터널이 하나라도 있으면 시나리오 계속
            if (t.IsFault)
            {
                allRepaired = false;
                break;
            }
        }

        if (allRepaired)
        {
            // 이 시나리오 종료 → 다음 시나리오 시작 시간 예약
            scenarioActive = false;
            nextScenarioStartTime = Time.time + delayBetweenScenarios;

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] 시나리오 고장 {currentFaults.Count}개 수리 완료 → {delayBetweenScenarios:F1}s 뒤 다음 시나리오 시작");
            }
        }
    }

    /// <summary>
    /// A/B/C 남은 개수가 모두 0이면 새 사이클로 초기화.
    /// </summary>
    private void ResetCycleCounts()
    {
        remainingA = cycleCountA;
        remainingB = cycleCountB;
        remainingC = cycleCountC;

        if (debugLogs)
        {
            int totalScenarios = remainingA + remainingB + remainingC;
            Debug.Log($"[RandomFailureScenario] 새 사이클 시작: A={remainingA}, B={remainingB}, C={remainingC}, 총 시나리오 수={totalScenarios}");
        }
    }

    /// <summary>
    /// 다음 시나리오에서 몇 개를 고장낼지(A/B/C 중 하나)를 결정한다.
    /// - A/B/C 중 아직 남아있는 타입들만 후보로 두고,
    ///   그 중에서 랜덤으로 하나를 선택.
    /// - 모든 타입의 remaining이 0이면 사이클 리셋 후 다시 선택.
    /// </summary>
    private int DecideFaultCountForNextScenario()
    {
        if (remainingA <= 0 && remainingB <= 0 && remainingC <= 0)
        {
            // 한 사이클을 다 썼으면 새 사이클로 리셋
            ResetCycleCounts();
        }

        // 남아 있는 타입들만 후보로 모은다.
        List<int> candidates = new List<int>(); // 3,2,1 을 담을 리스트

        if (remainingA > 0) candidates.Add(3); // A
        if (remainingB > 0) candidates.Add(2); // B
        if (remainingC > 0) candidates.Add(1); // C

        if (candidates.Count == 0)
        {
            // 이론상 올 수 없는 상황이지만 방어적으로 처리
            Debug.LogWarning("[RandomFailureScenario] 남은 시나리오 타입이 없습니다. 기본값 1개 고장 사용.");
            return 1;
        }

        int idx = Random.Range(0, candidates.Count);
        int faultsThisScenario = candidates[idx];

        // 선택된 타입에 따라 remaining 감소
        if (faultsThisScenario == 3)
        {
            remainingA--;
        }
        else if (faultsThisScenario == 2)
        {
            remainingB--;
        }
        else if (faultsThisScenario == 1)
        {
            remainingC--;
        }

        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] 다음 시나리오 타입 선택: faults={faultsThisScenario}, 남은 A/B/C={remainingA}/{remainingB}/{remainingC}");
        }

        return faultsThisScenario;
    }

    /// <summary>
    /// Returns a list of tunnels that are available for failure scenarios based on node filter settings.
    /// </summary>
    private List<TunnelController> GetAvailableTunnels()
    {
        if (!useNodeFilter)
        {
            // Return all tunnels when filter is disabled
            return new List<TunnelController>(allTunnels);
        }
        
        List<TunnelController> result = new List<TunnelController>();
        int enabledCount = 0;
        
        foreach (var tunnel in allTunnels)
        {
            if (tunnel == null) continue;
            
            // Find toggle for this nodeId
            NodeToggle toggle = null;
            foreach (var t in nodeToggles)
            {
                if (t.nodeId == tunnel.nodeId)
                {
                    toggle = t;
                    break;
                }
            }
            
            if (toggle != null)
            {
                if (toggle.isEnabled)
                {
                    result.Add(tunnel);
                    enabledCount++;
                }
            }
            else
            {
                // If tunnel not found in toggle list, include it by default
                result.Add(tunnel);
                
                if (debugLogs)
                {
                    Debug.LogWarning($"[RandomFailureScenario] 터널 nodeId={tunnel.nodeId}가 nodeToggles 목록에 없습니다. 기본적으로 포함합니다.");
                }
            }
        }
        
        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] {result.Count}/{allTunnels.Count}개 터널 사용 가능 (활성화된 노드: {enabledCount}개)");
        }
        
        return result;
    }

    void StartNewScenario()
    {
        // 이전 시나리오 정보 리셋
        currentFaults.Clear();

        // 이번 시나리오에서 몇 개를 고장낼지 결정 (A/B/C 중 하나)
        int faultsPerScenario = DecideFaultCountForNextScenario();

        // 필터링된 터널 목록 가져오기
        List<TunnelController> availableTunnels = GetAvailableTunnels();
        
        if (availableTunnels.Count == 0)
        {
            Debug.LogWarning("[RandomFailureScenario] 사용 가능한 터널이 없습니다. 시나리오를 건너뜁니다.");
            scenarioActive = false;
            nextScenarioStartTime = Time.time + delayBetweenScenarios;
            return;
        }

        if (availableTunnels.Count < faultsPerScenario)
        {
            if (debugLogs)
            {
                Debug.LogWarning($"[RandomFailureScenario] 필요한 고장 수({faultsPerScenario})보다 사용 가능한 터널({availableTunnels.Count})이 적습니다. 최대한 사용합니다.");
            }
            faultsPerScenario = availableTunnels.Count;
        }

        // 랜덤으로 섞기 (Fisher-Yates)
        List<TunnelController> shuffled = new List<TunnelController>(availableTunnels);
        int n = shuffled.Count;
        for (int i = 0; i < n; i++)
        {
            int j = Random.Range(i, n);
            var tmp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = tmp;
        }

        int count = Mathf.Min(faultsPerScenario, shuffled.Count);

        for (int i = 0; i < count; i++)
        {
            TunnelController t = shuffled[i];
            if (t == null) continue;

            // 자동 수리 끄기 (이 시나리오 동안은 로봇만 수리하게)
            t.autoRepairFixedDelay = false;

            // 이미 고장 상태면 한번 수리해두고 다시 고장낼 수도 있음
            if (t.IsFault)
            {
                t.ForceRepair();
            }

            // 이 터널을 고장 상태로 만들기
            t.ForceFail();

            currentFaults.Add(t);

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] 시나리오 고장 터널 선택: nodeId={t.nodeId}, name={t.name}");
            }
        }

        scenarioActive = true;

        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] 새 시나리오 시작: 고장 터널 {currentFaults.Count}개 (이번 타입 faultsPerScenario={faultsPerScenario}, 사용 가능 터널={availableTunnels.Count})");
        }
    }
}