using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Text;

using Random = UnityEngine.Random;

public class RepairTaskManager : MonoBehaviour
{
    [Header("debugRobotFlow")]
    [Tooltip("debugRobotFlow")]
    public bool debugRobotFlow = true;

   [Header("robots")]
    public List<AStarAgent> robots;   // was: public AStarAgent robot;

    private class RobotState {
        public AStarAgent robot;
        public RepairSite currentTarget;
        public bool busy = false;
        public System.Action onArrivedHandler; // stored so we can unsubscribe cleanly
        public RobotBattery battery;   // NEW
    }
    private List<RobotState> robotStates = new List<RobotState>();
    private bool dqnDecisionInProgress = false;
    public List<RepairSite> sites;

    [Header("Agent 2 - Robot Dispatcher")]
    public RobotDqnAgent robotDqnAgent;   // assign in Inspector; robots list must match `robots` order

    bool[] GetBusyFlags()
    {
        bool[] flags = new bool[robotStates.Count];
        for (int i = 0; i < robotStates.Count; i++)
            flags[i] = robotStates[i].busy;
        return flags;
    }

    [Header("dqnAgent")]
    [Tooltip("dqnAgent")]
    public DqnAgent dqnAgent;

    [Header("chooseRandomWhenMultiple")]
    [Tooltip("chooseRandomWhenMultiple")]
    public bool chooseRandomWhenMultiple = true;

    [Header("Local Policy")]
    [Tooltip("If true, use nearest (closest) site instead of random")]
    public bool chooseNearestWhenMultiple = false;

    [Header("useDqnSelection")]
    [Tooltip("useDqnSelection")]
    public bool useDqnSelection = true;

    [Range(0f, 1f)]
    [Tooltip("epsilon")]
    public float epsilon = 0.3f;

    [Tooltip("epsilonMin")]
    public float epsilonMin = 0.05f;

    [Tooltip("epsilonDecay")]
    public float epsilonDecay = 0.999f;

    [Header("Decision Timing")]
    [Tooltip("새 fault가 큐에 들어온 직후 action 요청 전 기다릴 프레임 수")]
    public int framesToWaitBeforeDecision = 1;

    [Header("wrongSelectionHandling")]
    [Tooltip("정상 노드를 잘못 선택했을 때, 도착 후 잠깐 확인하는 시간(TU)")]
    public int wrongSelectionInspectionTU = 1;


    [Tooltip("연속 오답 횟수. 정답 선택 시 0으로 초기화됨")]
    [SerializeField] private int consecutiveWrongSelectionCount = 0;

    // ==================== CSV LOGGING FOR BASELINES ====================
    [Header("CSV Logging (Baselines)")]
    [Tooltip("Enable CSV logging of episode stats (only when useDqnSelection = false)")]
    public bool logResultsToCSV = true;

    [Tooltip("File path relative to project root (e.g., 'baseline_results.csv')")]
    public string csvFilePath = "baseline_results.csv";

    [Tooltip("Maximum number of decision steps per episode (for logging)")]
    public int maxStepsPerEpisodeForLog = 50;

    [Tooltip("Current episode number (auto‑increments)")]
    [SerializeField] private int currentEpisodeForLog = 1;

    [Tooltip("Steps taken in current episode (for logging)")]
    [SerializeField] private int episodeStepCountLog = 0;

    [Tooltip("Accumulated reward in current episode (for logging)")]
    [SerializeField] private float episodeRewardLog = 0f;

    [Tooltip("Hits in current episode (for logging)")]
    [SerializeField] private int episodeHitCountLog = 0;

    // Throughput tracking for baseline
    private float episodeStartTimeLog = 0f;
    private int episodeStartExitsLog = 0;
    private bool isFirstStepOfEpisode = true;

    private StringBuilder csvBuffer;

    // ==================== END CSV LOGGING ====================

    RepairSite currentTarget;
    readonly List<RepairSite> pendingSites = new List<RepairSite>();
    bool robotBusy = false;
    bool decisionCoroutineRunning = false;
    bool reassignCoroutineQueued = false;

    // ----------------------------------------------------------------------
    // Existing methods (Awake, OnDestroy, Update, ScanSites, etc.)
    // ----------------------------------------------------------------------

    public int[] GetPendingCandidateNodeIds()
    {
        return GetAllCandidateNodeIds();
    }

    public int[] GetAllCandidateNodeIds()
    {
        if (sites == null || sites.Count == 0)
            return Array.Empty<int>();

        List<int> nodeIds = new List<int>(sites.Count);
        HashSet<int> seen = new HashSet<int>();

        foreach (var s in sites)
        {
            if (s == null || s.tunnel == null) continue;
            int nodeId = s.tunnel.nodeId;
            if (nodeId < 0) continue;
            if (seen.Add(nodeId))
                nodeIds.Add(nodeId);
        }

        return nodeIds.ToArray();
    }

    public int[] GetFaultyCandidateNodeIds()
    {
        List<int> faultyIds = new List<int>();
        foreach (var s in pendingSites)
        {
            if (s != null && s.tunnel != null)
                faultyIds.Add(s.tunnel.nodeId);
        }
        return faultyIds.ToArray();
    }

    void Awake()
    {
    robotStates.Clear();
    if (robots != null)
    {
        foreach (var r in robots)
        {
            if (r == null) continue;
            var rs = new RobotState { robot = r };
            rs.onArrivedHandler = () => HandleRobotArrived(rs);
            r.OnPathFinished += rs.onArrivedHandler;
            // NEW: battery wiring
            rs.battery = r.GetComponent<RobotBattery>();
            if (rs.battery != null)
                rs.battery.SetupParking(robotStates.Count);
            robotStates.Add(rs);
        }
    }

        // Initialize CSV logging
        if (logResultsToCSV)
        {
            // Delete old file to start fresh
            if (File.Exists(csvFilePath))
                File.Delete(csvFilePath);

            csvBuffer = new StringBuilder();
            // Updated header with throughput metrics
            csvBuffer.AppendLine("episode,steps,total_reward,hit_count,hit_ratio,episode_duration_sec,total_exits,episode_throughput,avg_reward");
            File.WriteAllText(csvFilePath, csvBuffer.ToString());
            csvBuffer.Clear();
            csvBuffer.AppendLine("episode,steps,total_reward,hit_count,hit_ratio,episode_duration_sec,total_exits,episode_throughput,avg_reward");

            if (debugRobotFlow)
                Debug.Log($"[RepairTaskManager] CSV logging enabled. File: {csvFilePath}");
        }
    }

    void OnDestroy()
    {
        foreach (var rs in robotStates)
        {
            if (rs.robot != null && rs.onArrivedHandler != null)
                rs.robot.OnPathFinished -= rs.onArrivedHandler;
        }

        // Flush any remaining CSV data
        if (logResultsToCSV && csvBuffer != null && csvBuffer.Length > 1)
        {
            File.AppendAllText(csvFilePath, csvBuffer.ToString());
        }
    }

    void Update()
    {
        if (robotStates == null || robotStates.Count == 0 || sites == null) return;
        ScanSites();
        CheckIdleRobotsForRecharge();
        TryAssignNextTask();
    }

    void CheckIdleRobotsForRecharge()
    {
        foreach (var rs in robotStates)
        {
            if (rs.busy || rs.battery == null) continue;
            if (!rs.battery.CanDispatch && !rs.battery.IsCharging)
            {
                StartCoroutine(GoRecharge(rs));
            }
        }
    }

    void ScanSites()
    {
        foreach (var s in sites)
        {
            if (s == null) continue;
            if (s.isQueued) continue;

            if (s.NeedsRepair)
            {
                s.isQueued = true;
                pendingSites.Add(s);

                if (debugRobotFlow)
                {
                    int nodeId = (s.tunnel != null) ? s.tunnel.nodeId : -1;
                    Debug.Log($"[RepairTaskManager] Fault queued. node={nodeId}, pendingCount={pendingSites.Count}");
                }
            }
        }
    }

    void TryAssignNextTask()
    {
        if (dqnDecisionInProgress) return;
        if (pendingSites.Count == 0) return;
        if (!HasIdleRobot()) return;

        bool canUseDqn =
            useDqnSelection &&
            dqnAgent != null &&
            dqnAgent.tcpClient != null &&
            dqnAgent.tcpClient.IsConnected;

        if (canUseDqn)
        {
            List<int> candidateNodeIds = new List<int>(GetFaultyCandidateNodeIds());

            if (candidateNodeIds.Count == 0)
            {
                if (dqnAgent.debugLogs)
                    Debug.LogWarning("[RepairTaskManager] No valid candidate node ids. Fallback to local policy.");
                PickAndAssignLocalPolicy();
                return;
            }

            StartCoroutine(CoRequestDecisionAfterSync(candidateNodeIds));
        }
        else
        {
            PickAndAssignLocalPolicy();
        }
    }

    // NEW helper -- add this as a new method anywhere in the class
    bool HasIdleRobot()
    {
        foreach (var rs in robotStates)
            if (!rs.busy && (rs.battery == null || rs.battery.CanDispatch))
                return true;
        return false;
    }

    // NEW helper -- this is the "nearest robot" logic. Add anywhere in the class.
    RobotState FindNearestIdleRobot(Vector3 targetPos)
    {
        RobotState best = null;
        float bestDist = float.MaxValue;

        foreach (var rs in robotStates)
        {
            if (rs.busy || rs.robot == null) continue;
            if (rs.battery != null && !rs.battery.CanDispatch) continue;   // NEW

            Vector3 p = rs.robot.transform.position;
            p.y = 0f;
            float d = Vector3.Distance(p, targetPos);

            if (d < bestDist)
            {
                bestDist = d;
                best = rs;
            }
        }
        return best;
    }

    IEnumerator CoRequestDecisionAfterSync(List<int> candidateNodeIds)
    {
        if (dqnDecisionInProgress)
            yield break;

        dqnDecisionInProgress = true;

        int waitFrames = Mathf.Max(0, framesToWaitBeforeDecision);
        for (int i = 0; i < waitFrames; i++)
            yield return null;

        if (FactoryEnvManager.Instance != null)
            FactoryEnvManager.Instance.SyncNodeStatesImmediate();

        if (dqnAgent != null)
            dqnAgent.ForceRefreshSnapshotsNow();

        yield return StartCoroutine(dqnAgent.CoRequestActionAndPickNode(
            candidateNodeIds,
            epsilon,
            (chosenNodeId, isRandomFromEps, capturedState) =>
            {
                epsilon = Mathf.Max(epsilonMin, epsilon * epsilonDecay);

                if (chosenNodeId < 0)
                {
                    dqnDecisionInProgress = false; // release -- no further async step follows
                    if (dqnAgent.debugLogs)
                        Debug.LogWarning("[RepairTaskManager] DQN returned invalid node. Fallback to local policy.");
                    PickAndAssignLocalPolicy();
                    return;
                }

                RepairSite chosenSite = FindSiteByNodeId(chosenNodeId);
                if (chosenSite == null)
                {
                    dqnDecisionInProgress = false; // release -- no further async step follows
                    if (dqnAgent.debugLogs)
                        Debug.LogWarning($"[RepairTaskManager] chosenNodeId={chosenNodeId} not found in sites. Fallback to local policy.");
                    PickAndAssignLocalPolicy();
                    return;
                }

                // dqnDecisionInProgress stays TRUE -- released only once AssignSiteToNearestRobot's
                // full pipeline (including Agent 2's async robot-selection round trip) commits.
                AssignSiteToNearestRobot(chosenSite, chosenNodeId, capturedState);
            }));
    }

    void AssignSiteToNearestRobot(RepairSite site, int nodeId, float[] capturedState = null)
    {
        Vector3 targetPos = (site.RepairPoint != null) ? site.RepairPoint.position : Vector3.zero;
        int repairTU = (site.tunnel != null) ? Mathf.Max(0, site.tunnel.repairDurationTU) : 0;

        bool useAgent2 = robotDqnAgent != null;

        if (!useAgent2)
        {
            RobotState rsFallback = FindNearestIdleRobot(targetPos);
            if (rsFallback == null)
            {
                dqnDecisionInProgress = false;
                return;
            }
            RemovePendingSiteIfExists(site);
            if (dqnAgent != null) dqnAgent.RecordAction(nodeId, nodeId, capturedState);
            rsFallback.currentTarget = site;
            MoveRobotToCurrentTarget(rsFallback);
            dqnDecisionInProgress = false;
            return;
        }

        StartCoroutine(CoDispatchViaAgent2(site, nodeId, targetPos, repairTU, capturedState));
    }

    IEnumerator CoDispatchViaAgent2(RepairSite site, int nodeId, Vector3 targetPos, int repairTU, float[] capturedState)
    {
        bool[] busyFlags = GetBusyFlags();

        yield return StartCoroutine(robotDqnAgent.CoPickRobot(
            busyFlags, targetPos, nodeId, repairTU,
            (chosenIdx, isRandom) =>
            {
                if (chosenIdx < 0 || chosenIdx >= robotStates.Count)
                {
                    if (debugRobotFlow) Debug.LogWarning("[RepairTaskManager] Agent2 returned invalid robot index. Skipping.");
                    dqnDecisionInProgress = false;
                    return;
                }

                RobotState rs = robotStates[chosenIdx];
                RemovePendingSiteIfExists(site);
                if (dqnAgent != null) dqnAgent.RecordAction(nodeId, nodeId, capturedState);

                rs.currentTarget = site;
                MoveRobotToCurrentTarget(rs);
                dqnDecisionInProgress = false;
            }));
    }

    void PickAndAssignLocalPolicy()
    {
        if (pendingSites.Count == 0) return;
        if (!HasIdleRobot()) return;

        RepairSite chosenSite = null;
        RobotState chosenRobot = null;

        if (chooseNearestWhenMultiple)
        {
            float bestDist = float.MaxValue;

            foreach (var rs in robotStates)
            {
                if (rs.busy || rs.robot == null) continue;

                Vector3 robotPos = rs.robot.transform.position;
                robotPos.y = 0f;

                foreach (var site in pendingSites)
                {
                    if (site == null || site.RepairPoint == null) continue;

                    Vector3 targetPos = site.RepairPoint.position;
                    targetPos.y = 0f;
                    float dist = Vector3.Distance(robotPos, targetPos);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        chosenSite = site;
                        chosenRobot = rs;
                    }
                }
            }
        }

        // Fallback: random/first site, then nearest idle robot to it
        if (chosenSite == null)
        {
            chosenSite = (chooseRandomWhenMultiple && pendingSites.Count > 1)
                ? pendingSites[Random.Range(0, pendingSites.Count)]
                : pendingSites[0];

            Vector3 pos = (chosenSite.RepairPoint != null) ? chosenSite.RepairPoint.position : Vector3.zero;
            chosenRobot = FindNearestIdleRobot(pos);
        }

        if (chosenSite == null || chosenRobot == null) return;

        RemovePendingSiteIfExists(chosenSite);

        if (dqnAgent != null)
        {
            int nodeId = (chosenSite.tunnel != null) ? chosenSite.tunnel.nodeId : -1;
            dqnAgent.RecordAction(nodeId, nodeId);
        }

        chosenRobot.currentTarget = chosenSite;
        MoveRobotToCurrentTarget(chosenRobot);
    }

    void HandleRobotArrived(RobotState rs)
    {
        rs.busy = true;

        if (rs.currentTarget == null)
        {
            rs.busy = false;
            QueueReassignNextFrame();
            return;
        }

        if (rs.currentTarget.NeedsRepair)
            StartCoroutine(CoRepairCurrentTarget(rs, rs.currentTarget));
        else
            StartCoroutine(CoInspectNormalTarget(rs, rs.currentTarget));
    }

    void MoveRobotToCurrentTarget(RobotState rs)
    {
        if (rs.robot == null || rs.currentTarget == null || rs.currentTarget.RepairPoint == null)
        {
            rs.busy = false;
            return;
        }

        Vector3 robotPos = rs.robot.transform.position;
        Vector3 targetPos = rs.currentTarget.RepairPoint.position;
        robotPos.y = 0f;
        targetPos.y = 0f;

        float dist = Vector3.Distance(robotPos, targetPos);
        int nodeId = (rs.currentTarget.tunnel != null) ? rs.currentTarget.tunnel.nodeId : -1;

        if (debugRobotFlow)
            Debug.Log($"[RepairTaskManager] Move start -> node={nodeId}, needsRepairNow={rs.currentTarget.NeedsRepair}, dist={dist:F3}");

        const float immediateHandleThreshold = 0.05f;
        if (dist <= immediateHandleThreshold)
        {
            rs.busy = true;
            HandleRobotArrived(rs);
            return;
        }

        const float arriveThreshold = 0.6f;
        if (dist <= arriveThreshold)
        {
            HandleRobotArrived(rs);
            return;
        }

        rs.robot.SetTarget(rs.currentTarget.RepairPoint, true);
        rs.busy = true;
    }

    IEnumerator CoRepairCurrentTarget(RobotState rs, RepairSite site)
    {
        if (site == null)
        {
            rs.currentTarget = null;
            rs.busy = false;
            QueueReassignNextFrame();
            yield break;
        }

        int nodeId = (site.tunnel != null) ? site.tunnel.nodeId : -1;
        site.BeginRepairVisual();

        float sPerTU = Mathf.Max(1e-6f, FactoryEnvManager.Instance != null ? FactoryEnvManager.Instance.secondsPerTU : 1f);
        int repairTU = (site.tunnel != null) ? Mathf.Max(0, site.tunnel.repairDurationTU) : 0;

        if (dqnAgent != null)
            dqnAgent.SetSelectionOutcome(nodeId, true);

        if (!useDqnSelection && logResultsToCSV)
            RecordStepForBaseline(true);

        if (FactoryEnvManager.Instance != null)
            FactoryEnvManager.Instance.NotifyRobotArrivedAtDecisionTarget(nodeId, repairTU);
        // NEW: drain battery using the SAME moveTU/repairTU the reward will use.
        if (rs.battery != null && FactoryEnvManager.Instance != null &&
            FactoryEnvManager.Instance.TryPeekEnergyBreakdown(nodeId, out int moveTU, out int repairTUForBattery))
        {
            rs.battery.ConsumeMoveEnergyByTU(moveTU);
            // Repair energy is drained after the wait loop below (see step 2f),
            // reusing repairTUForBattery captured here.
        }

        float wait = repairTU * sPerTU;
        float elapsed = 0f;
        while (elapsed < wait)
        {
            elapsed += Time.deltaTime;
            if (wait > 0f)
            {
                float progress = Mathf.Clamp01(elapsed / wait);
                site.UpdateRepairVisual(progress);
            }
            yield return null;
        }

        site.OnRepaired();
        site.EndRepairVisual();
        site.isQueued = false;

        if (FactoryEnvManager.Instance != null)
            FactoryEnvManager.Instance.SyncNodeStatesImmediate();

        // NEW: capture Agent 2's move/repair TU BEFORE Agent 1 consumes/removes this
        // decision snapshot inside FinishStepAndSend -> ComputeGlobalReward.
        bool hasAgent2Breakdown = false;
        int finalMoveTU = 0, finalRepairTU = 0;
        if (robotDqnAgent != null && FactoryEnvManager.Instance != null)
        {
            hasAgent2Breakdown = FactoryEnvManager.Instance.TryPeekEnergyBreakdown(nodeId, out finalMoveTU, out finalRepairTU);
        }

        if (dqnAgent != null && site.tunnel != null)
        {
            dqnAgent.ForceRefreshSnapshotsNow();
            dqnAgent.FinishStepAndSend(nodeId); // removes _pendingDecisions[nodeId] -- must run AFTER the peek above
        }

        // NEW: drain repair energy now that the work is actually done.
        if (rs.battery != null)
            rs.battery.ConsumeRepairEnergyByTU(repairTU);

        // NEW: close out Agent 2's transition, using the values captured before Agent 1 consumed them.
        if (robotDqnAgent != null)
        {
            if (hasAgent2Breakdown)
            {
                robotDqnAgent.FinishRobotStep(nodeId, finalMoveTU, finalRepairTU, GetBusyFlags());
            }
            else if (debugRobotFlow)
            {
                Debug.LogWarning($"[RepairTaskManager] Could not close Agent2 transition for node={nodeId}: TryPeekEnergyBreakdown returned false.");
            }
        }

        rs.currentTarget = null;

        if (rs.battery != null && !rs.battery.CanDispatch)
        {
            StartCoroutine(GoRecharge(rs));
        }
        else
        {
            rs.busy = false;
            QueueReassignNextFrame();
        }
    }

    IEnumerator CoInspectNormalTarget(RobotState rs, RepairSite site)
    {
        if (site == null)
        {
            rs.currentTarget = null;
            rs.busy = false;
            QueueReassignNextFrame();
            yield break;
        }

        int nodeId = (site.tunnel != null) ? site.tunnel.nodeId : -1;
        int inspectTU = Mathf.Max(0, wrongSelectionInspectionTU);

        if (debugRobotFlow)
            Debug.Log($"[RepairTaskManager] Arrived at NORMAL node={nodeId}. Inspect for {inspectTU} TU.");

        if (FactoryEnvManager.Instance != null)
            FactoryEnvManager.Instance.NotifyRobotArrivedAtDecisionTarget(nodeId, inspectTU);
        // NEW: drain battery using the SAME moveTU/repairTU the reward will use.
        if (rs.battery != null && FactoryEnvManager.Instance != null &&
            FactoryEnvManager.Instance.TryPeekEnergyBreakdown(nodeId, out int moveTU, out int repairTUForBattery))
        {
            rs.battery.ConsumeMoveEnergyByTU(moveTU);
            // Repair energy is drained after the wait loop below (see step 2f),
            // reusing repairTUForBattery captured here.
        }


        float wait = (FactoryEnvManager.Instance != null)
            ? FactoryEnvManager.Instance.TUToSeconds(inspectTU)
            : inspectTU;

        float elapsed = 0f;
        while (elapsed < wait)
        {
            elapsed += Time.deltaTime;
            yield return null;
        }

        if (dqnAgent != null)
            dqnAgent.SetSelectionOutcome(nodeId, true);

        

        if (!useDqnSelection && logResultsToCSV)
            RecordStepForBaseline(false);

        if (FactoryEnvManager.Instance != null)
            FactoryEnvManager.Instance.SyncNodeStatesImmediate();

        // NEW: capture Agent 2's move/repair TU BEFORE Agent 1 consumes/removes this
        // decision snapshot inside FinishStepAndSend -> ComputeGlobalReward.
        bool hasAgent2Breakdown = false;
        int finalMoveTU = 0, finalRepairTU = 0;
        if (robotDqnAgent != null && FactoryEnvManager.Instance != null)
        {
            hasAgent2Breakdown = FactoryEnvManager.Instance.TryPeekEnergyBreakdown(nodeId, out finalMoveTU, out finalRepairTU);
        }

        if (dqnAgent != null && site.tunnel != null)
        {
            dqnAgent.ForceRefreshSnapshotsNow();
            dqnAgent.FinishStepAndSend(nodeId); // removes _pendingDecisions[nodeId] -- must run AFTER the peek above
        }

        // NEW: drain repair energy now that the work is actually done.
        if (rs.battery != null)
            rs.battery.ConsumeRepairEnergyByTU(inspectTU);

        // NEW: close out Agent 2's transition, using the values captured before Agent 1 consumed them.
        if (robotDqnAgent != null)
        {
            if (hasAgent2Breakdown)
            {
                robotDqnAgent.FinishRobotStep(nodeId, finalMoveTU, finalRepairTU, GetBusyFlags());
            }
            else if (debugRobotFlow)
            {
                Debug.LogWarning($"[RepairTaskManager] Could not close Agent2 transition for node={nodeId}: TryPeekEnergyBreakdown returned false.");
            }
        }

        rs.currentTarget = null;

        if (rs.battery != null && rs.battery.IsLow)
        {
            StartCoroutine(GoRecharge(rs));
        }
        else
        {
            rs.busy = false;
            QueueReassignNextFrame();
        }
    }

    void QueueReassignNextFrame()
    {
        if (reassignCoroutineQueued)
            return;

        StartCoroutine(CoReassignNextFrame());
    }

    IEnumerator CoReassignNextFrame()
    {
        reassignCoroutineQueued = true;
        yield return null;
        reassignCoroutineQueued = false;
        TryAssignNextTask();
    }

    RepairSite FindSiteByNodeId(int nodeId)
    {
        if (sites == null) return null;

        foreach (var s in sites)
        {
            if (s == null || s.tunnel == null) continue;
            if (s.tunnel.nodeId == nodeId)
                return s;
        }

        return null;
    }

    void RemovePendingSiteIfExists(RepairSite site)
    {
        if (site == null) return;

        for (int i = 0; i < pendingSites.Count; i++)
        {
            if (ReferenceEquals(pendingSites[i], site))
            {
                pendingSites.RemoveAt(i);
                return;
            }
        }
    }

    IEnumerator GoRecharge(RobotState rs)
    {
        rs.busy = true; // keep excluded from the idle pool while charging

        if (rs.battery != null && rs.battery.chargingStation != null && rs.robot != null)
        {
            if (debugRobotFlow)
                Debug.Log($"[RepairTaskManager] {rs.robot.name} low battery ({rs.battery.EnergyPercent:P0}), returning to charge.");

            rs.robot.SetTarget(rs.battery.chargingStation, true);
            while (rs.robot.IsMoving)
                yield return null;
        }

        if (rs.battery != null)
            yield return StartCoroutine(rs.battery.ChargeUntilDispatchable());

        rs.busy = false;
        QueueReassignNextFrame();
    }

    // ==================== CSV LOGGING METHODS ====================
    private void RecordStepForBaseline(bool isHit)
    {
        // Get the reward for this step from FactoryEnvManager (the total reward for the decision step)
        float stepReward = 0f;
        if (FactoryEnvManager.Instance != null)
            stepReward = FactoryEnvManager.Instance.GetLastGlobalReward();

        // Capture start time and exits on first step of episode
        if (isFirstStepOfEpisode)
        {
            isFirstStepOfEpisode = false;
            if (FactoryEnvManager.Instance != null)
            {
                episodeStartTimeLog = FactoryEnvManager.Instance.GetSimulationTimeSeconds();
                episodeStartExitsLog = FactoryEnvManager.Instance.TotalExits;
                if (debugRobotFlow)
                {
                    Debug.Log($"[CSV] Episode {currentEpisodeForLog} start: time={episodeStartTimeLog:F2}, exits={episodeStartExitsLog}");
                }
            }
        }

        episodeStepCountLog++;
        episodeRewardLog += stepReward;
        if (isHit) episodeHitCountLog++;

        if (debugRobotFlow)
        {
            Debug.Log($"[CSV] Step {episodeStepCountLog}/{maxStepsPerEpisodeForLog} | hit={isHit} | stepReward={stepReward:F3} | totalReward={episodeRewardLog:F3}");
        }

        // Episode end condition
        if (episodeStepCountLog >= maxStepsPerEpisodeForLog)
        {
            LogEpisodeResultForBaseline();
        }
    }

    private void LogEpisodeResultForBaseline()
    {
        if (!logResultsToCSV) return;
        if (episodeStepCountLog == 0) return;

        float hitRatio = (float)episodeHitCountLog / episodeStepCountLog;
        float avgReward = episodeRewardLog / episodeStepCountLog;

        // Get episode end time and exits
        float episodeEndTime = 0f;
        int episodeEndExits = 0;
        if (FactoryEnvManager.Instance != null)
        {
            episodeEndTime = FactoryEnvManager.Instance.GetSimulationTimeSeconds();
            episodeEndExits = FactoryEnvManager.Instance.TotalExits;
        }

        float episodeDuration = episodeEndTime - episodeStartTimeLog;
        int episodeTotalExits = episodeEndExits - episodeStartExitsLog;
        float episodeThroughput = episodeDuration > 0f ? episodeTotalExits / episodeDuration : 0f;

        // Format CSV line
        string line = $"{currentEpisodeForLog},{episodeStepCountLog},{episodeRewardLog:F3},{episodeHitCountLog},{hitRatio:F3},{episodeDuration:F3},{episodeTotalExits},{episodeThroughput:F3},{avgReward:F3}";
        csvBuffer.AppendLine(line);

        // Write to file immediately
        File.AppendAllText(csvFilePath, line + "\n");

        if (debugRobotFlow)
        {
            Debug.Log($"[CSV] Episode {currentEpisodeForLog} logged: steps={episodeStepCountLog}, reward={episodeRewardLog:F3}, hits={episodeHitCountLog}, hitRatio={hitRatio:F3}, duration={episodeDuration:F2}s, exits={episodeTotalExits}, throughput={episodeThroughput:F3}, avgReward={avgReward:F3}");
        }

        // Reset episode counters
        currentEpisodeForLog++;
        episodeStepCountLog = 0;
        episodeRewardLog = 0f;
        episodeHitCountLog = 0;
        isFirstStepOfEpisode = true;
        episodeStartTimeLog = 0f;
        episodeStartExitsLog = 0;
    }
    // ==================== END CSV LOGGING ====================
}