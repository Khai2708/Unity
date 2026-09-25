// RobotDqnAgent.cs
// Agent 2: selects WHICH idle robot handles a chosen fault node.
//
// State: see BuildStateVector -- maxRobots*4 + 2 features (dist, battery%,
// busy, survivability per robot + repairTU_norm, idle_count_norm context).
//
// Reward (composite, per user spec):
//   R2 = -(w_move*moveCostNorm + w_repair*repairCostNorm)
//        - w_fail*failurePenaltyRaw
//        - w_fleet*fleetImbalanceNorm
//
// All cost terms reuse FactoryEnvManager's moveEnergyPerTU / repairEnergyPerTU /
// maxMoveTU / maxRepairTU -- single source of truth, same as RobotBattery.

using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class RobotDqnAgent : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────
    [Header("TCP")]
    public RobotDqnTcpClient tcpClient;

    [Header("Robots (same list & order as RepairTaskManager's robots)")]
    public List<AStarAgent> robots;
    [Tooltip("Fixed capacity for the state/action vectors. Must equal Python's MAX_ROBOTS. " +
             "Set this to the largest fleet size you'll ever test, NOT the current robot count.")]
    public int maxRobots = 4;

    [Header("Normalisation")]
    [Tooltip("Fallback robot speed (m/s), used only to estimate move-TU for the survivability feature.")]
    public float assumedRobotSpeed = 3.5f;

    [Header("Reward Weights")]
    public float wMove = 0.4f;
    public float wRepair = 0.4f;
    public float wFail = 2.0f;
    public float wFleet = 0.2f;

    [Header("Timeout")]
    public float actionTimeoutSec = 10f;

    [Header("Debug")]
    public bool debugLogs = true;

    private RobotBattery[] _batteries;
    private FactoryEnvManager _env;

    // ── DTOs ────────────────────────────────────────────────────────────────
    [Serializable]
    public class RobotActionRequest
    {
        public string type = "robot_action_request";
        public float[] state;
        public int[] idle_robot_indices;
        public float epsilon = -1f;
    }

    [Serializable]
    public class RobotTransition
    {
        public string type = "robot_transition";
        public int robot_index;
        public int node_id;
        public float reward;

        // backward-compatible simple fields
        public float travel_dist_norm;
        public float battery_drain_norm;

        // composite reward breakdown
        public float move_cost_norm;
        public float repair_cost_norm;
        public float fleet_balance_norm;
        public float move_contrib;
        public float repair_contrib;
        public float fleet_contrib;
        public float failure_penalty;
        public float mission_success;
        public int module_count;

        public float[] state_t;
        public float[] state_tp1;
        public int[] idle_robots_tp1;
    }

    private class PendingRobotTransition
    {
        public int robotIndex;
        public int nodeId;
        public float[] stateT;
        public float batteryBeforePct; // chosen robot's battery% at dispatch time
        public float distanceAtDispatch;
        public float recordTime;
    }

    private readonly Dictionary<int, PendingRobotTransition> _pending
        = new Dictionary<int, PendingRobotTransition>();

    private bool _waitingReply = false;
    private bool _hasReply = false;
    private int _repliedRobotIndex = -1;
    private bool _replyIsRandom = false;
    private float[] _replyQValues = null;

    // ── lifecycle ───────────────────────────────────────────────────────────
    void Awake()
    {
        if (tcpClient != null)
            tcpClient.OnRobotActionReply += HandleReply;
        _env = FactoryEnvManager.Instance;
    }

    void Start()
    {
        if (_env == null) _env = FactoryEnvManager.Instance;
        CacheBatteries();
    }

    void OnDestroy()
    {
        if (tcpClient != null)
            tcpClient.OnRobotActionReply -= HandleReply;
    }

    void CacheBatteries()
    {
        if (robots == null) { _batteries = new RobotBattery[0]; return; }
        _batteries = new RobotBattery[robots.Count];
        for (int i = 0; i < robots.Count; i++)
        {
            if (robots[i] == null) continue;
            _batteries[i] = robots[i].GetComponent<RobotBattery>();
            if (_batteries[i] == null && debugLogs)
                Debug.LogWarning($"[RobotDqnAgent] robot '{robots[i].name}' has no RobotBattery component.");
        }
    }

    // ── energy helpers (single source of truth: FactoryEnvManager) ─────────
    float MoveEnergyPerTU => _env != null ? _env.moveEnergyPerTU : 1f;
    float RepairEnergyPerTU => _env != null ? _env.repairEnergyPerTU : 1f;
    int MaxMoveTU => _env != null ? Mathf.Max(1, _env.maxMoveTU) : 1;
    int MaxRepairTU => _env != null ? Mathf.Max(1, _env.maxRepairTU) : 1;
    float SecondsPerTU => _env != null ? Mathf.Max(1e-6f, _env.secondsPerTU) : 1f;
    float MaxTravelDistance => Mathf.Max(1f, assumedRobotSpeed * SecondsPerTU * MaxMoveTU);

    float EstimateMoveTU(float distanceMeters)
    {
        float speed = Mathf.Max(0.01f, assumedRobotSpeed);
        return (distanceMeters / speed) / SecondsPerTU;
    }

    float EstimateJobCost(float distanceMeters, int repairTU)
    {
        float moveTU = EstimateMoveTU(distanceMeters);
        return (moveTU * MoveEnergyPerTU) + (Mathf.Max(0, repairTU) * RepairEnergyPerTU);
    }

    // ── public API ──────────────────────────────────────────────────────────

    public IEnumerator CoPickRobot(
        bool[] busyFlags,
        Vector3 targetPos,
        int nodeId,
        int repairTU,
        Action<int, bool> onDone)
    {
        bool tcpAvailable = tcpClient != null && tcpClient.IsConnected;

        List<int> idleIndices = GetIdleRobotIndices(busyFlags);
        if (idleIndices.Count == 0)
        {
            onDone?.Invoke(-1, true);
            yield break;
        }

        if (idleIndices.Count == 1)
        {
            int only = idleIndices[0];
            RecordDirectDispatch(only, nodeId, busyFlags, targetPos, repairTU);
            onDone?.Invoke(only, false);
            yield break;
        }

        if (!tcpAvailable)
        {
            int nearest = NearestIdle(idleIndices, targetPos);
            RecordDirectDispatch(nearest, nodeId, busyFlags, targetPos, repairTU);
            onDone?.Invoke(nearest, true);
            yield break;
        }

        float[] stateVec = BuildStateVector(busyFlags, targetPos, nodeId, repairTU);

        var req = new RobotActionRequest
        {
            type = "robot_action_request",
            state = stateVec,
            idle_robot_indices = idleIndices.ToArray(),
            epsilon = -1f
        };

        string json = JsonUtility.ToJson(req);
        _waitingReply = false;
        _hasReply = false;
        _waitingReply = true;
        tcpClient.SendJsonLine(json);

        if (debugLogs)
            Debug.Log($"[RobotDqnAgent] robot_action_request sent. idle=[{string.Join(",", idleIndices)}] node={nodeId} repairTU={repairTU}");

        float startTime = Time.realtimeSinceStartup;
        while (_waitingReply && Time.realtimeSinceStartup - startTime < actionTimeoutSec)
            yield return null;

        int chosen;
        bool isRandom;

        if (!_hasReply)
        {
            if (debugLogs) Debug.LogWarning("[RobotDqnAgent] Timeout. Fallback to nearest.");
            chosen = NearestIdle(idleIndices, targetPos);
            isRandom = true;
        }
        else
        {
            chosen = _repliedRobotIndex;
            isRandom = _replyIsRandom;

            bool chosenIsValidIdle = chosen >= 0 && chosen < busyFlags.Length && !busyFlags[chosen] &&
                                      (_batteries == null || chosen >= _batteries.Length || _batteries[chosen] == null || _batteries[chosen].CanDispatch);

            if (!chosenIsValidIdle)
            {
                if (debugLogs) Debug.LogWarning($"[RobotDqnAgent] Chosen robot {chosen} not idle/dispatchable. Fallback (not exploration).");
                chosen = NearestIdle(idleIndices, targetPos);
                // deliberately not forcing isRandom=true here -- see prior note on
                // not conflating epsilon exploration with constraint-violation fallback.
            }
        }

        RecordDirectDispatch(chosen, nodeId, busyFlags, targetPos, repairTU, stateVec);
        onDone?.Invoke(chosen, isRandom);
    }

    /// <summary>
    /// Call when the chosen robot completes (or fails) its mission.
    /// moveTU/repairTU should come from FactoryEnvManager.TryPeekEnergyBreakdown(nodeId, ...)
    /// -- the SAME numbers the production reward uses, so Agent 2's cost terms
    /// never drift from Agent 1's reward or the robot's actual battery drain.
    /// </summary>
    public void FinishRobotStep(int nodeId, int moveTU, int repairTU, bool[] busyFlagsAfter)
    {
        if (!_pending.TryGetValue(nodeId, out var p))
        {
            if (debugLogs) Debug.LogWarning($"[RobotDqnAgent] FinishRobotStep: no pending for node {nodeId}");
            return;
        }
        _pending.Remove(nodeId);

        if (tcpClient == null || !tcpClient.IsConnected)
        {
            if (debugLogs) Debug.LogWarning($"[RobotDqnAgent] FinishRobotStep: tcpClient unavailable for node {nodeId}, transition dropped silently before this fix.");
            return;
        }

        RobotBattery chosenBattery = (_batteries != null && p.robotIndex < _batteries.Length) ? _batteries[p.robotIndex] : null;
        float batteryAfterPct = chosenBattery != null ? chosenBattery.EnergyPercent : 1f;

        // ── cost terms (TU-based, same constants as RobotBattery/reward) ──
        float moveCostNorm = Mathf.Clamp01((float)Mathf.Max(0, moveTU) / MaxMoveTU);
        float repairCostNorm = Mathf.Clamp01((float)Mathf.Max(0, repairTU) / MaxRepairTU);

        // ── failure proxy: finished the job at 0% battery ──
        // NOTE: the sim doesn't currently hard-stop a robot for running out of
        // battery mid-mission, so this is a soft/post-hoc signal, not a true
        // stranding event. Revisit if you add a real abort-on-empty mechanic.
        bool missionFailed = chosenBattery != null && !chosenBattery.unlimitedEnergy && batteryAfterPct <= 0f;
        float failurePenaltyRaw = missionFailed ? 1f : 0f;

        // ── fleet balance: |chosen robot battery% - fleet avg battery%| ──
        float fleetAvgPct = GetFleetAverageBatteryPct();
        float fleetImbalanceNorm = Mathf.Clamp01(Mathf.Abs(batteryAfterPct - fleetAvgPct));

        // ── weighted contributions (signed, for logging/analysis) ──
        float moveContrib = -wMove * moveCostNorm;
        float repairContrib = -wRepair * repairCostNorm;
        float fleetContrib = -wFleet * fleetImbalanceNorm;
        float failurePenaltyContrib = -wFail * failurePenaltyRaw;

        float reward = moveContrib + repairContrib + fleetContrib + failurePenaltyContrib;

        // backward-compatible simple fields
        float distNorm = Mathf.Clamp01(p.distanceAtDispatch / MaxTravelDistance);
        float battDrainNorm = Mathf.Clamp01(p.batteryBeforePct - batteryAfterPct);

        float[] stateT1 = BuildStateVectorCurrentOnly(busyFlagsAfter);
        int[] idleNext = GetIdleRobotIndices(busyFlagsAfter).ToArray();

        var msg = new RobotTransition
        {
            type = "robot_transition",
            robot_index = p.robotIndex,
            node_id = nodeId,
            reward = reward,

            travel_dist_norm = distNorm,
            battery_drain_norm = battDrainNorm,

            move_cost_norm = moveCostNorm,
            repair_cost_norm = repairCostNorm,
            fleet_balance_norm = fleetImbalanceNorm,
            move_contrib = moveContrib,
            repair_contrib = repairContrib,
            fleet_contrib = fleetContrib,
            failure_penalty = failurePenaltyContrib,
            mission_success = missionFailed ? 0f : 1f,
            module_count = 1, // placeholder -- unused in this project's semantics

            state_t = p.stateT,
            state_tp1 = stateT1,
            idle_robots_tp1 = idleNext
        };

        tcpClient.SendJsonLine(JsonUtility.ToJson(msg));

        if (debugLogs)
            Debug.Log($"[RobotDqnAgent] robot_transition node={nodeId} robot={p.robotIndex} " +
                      $"r={reward:F3} (move={moveContrib:F3} repair={repairContrib:F3} " +
                      $"fleet={fleetContrib:F3} fail={failurePenaltyContrib:F3}) success={!missionFailed}");
    }

    float GetFleetAverageBatteryPct()
    {
        if (_batteries == null || _batteries.Length == 0) return 1f;
        float sum = 0f;
        int count = 0;
        foreach (var b in _batteries)
        {
            if (b == null) continue;
            sum += b.EnergyPercent;
            count++;
        }
        return count > 0 ? sum / count : 1f;
    }

    // ── state building ──────────────────────────────────────────────────────

    float[] BuildStateVector(bool[] busyFlags, Vector3 targetPos, int nodeId, int repairTU)
    {
        int numFeatures = maxRobots * 4 + 2;
        float[] s = new float[numFeatures];

        for (int i = 0; i < maxRobots; i++)
        {
            int baseIdx = i * 4;
            if (robots != null && i < robots.Count && robots[i] != null)
            {
                AStarAgent agent = robots[i];
                RobotBattery battery = (_batteries != null && i < _batteries.Length) ? _batteries[i] : null;

                float dist = Vector3.Distance(agent.transform.position, targetPos);
                float distNorm = Mathf.Clamp01(dist / MaxTravelDistance);
                float energyPercent = battery != null ? battery.EnergyPercent : 1f;
                bool busy = (busyFlags != null && i < busyFlags.Length) ? busyFlags[i] : false;

                float currentEnergy = battery != null ? battery.CurrentEnergy : 1f;
                float maxEnergy = battery != null ? Mathf.Max(1f, battery.maxEnergy) : 1f;
                float jobCost = battery != null && !battery.unlimitedEnergy ? EstimateJobCost(dist, repairTU) : 0f;
                float survivability = battery != null && !battery.unlimitedEnergy
                    ? Mathf.Clamp01((currentEnergy - jobCost) / maxEnergy)
                    : 1f;

                s[baseIdx + 0] = distNorm;
                s[baseIdx + 1] = energyPercent;
                s[baseIdx + 2] = busy ? 1f : 0f;
                s[baseIdx + 3] = survivability;
            }
            else
            {
                s[baseIdx + 0] = 1f;
                s[baseIdx + 1] = 0f;
                s[baseIdx + 2] = 1f;
                s[baseIdx + 3] = 0f;
            }
        }

        int ctx = maxRobots * 4;
        s[ctx + 0] = Mathf.Clamp01((float)repairTU / MaxRepairTU);
        s[ctx + 1] = Mathf.Clamp01(GetIdleCount(busyFlags) / (float)Mathf.Max(1, maxRobots));

        return s;
    }

    float[] BuildStateVectorCurrentOnly(bool[] busyFlags)
    {
        int numFeatures = maxRobots * 4 + 2;
        float[] s = new float[numFeatures];

        for (int i = 0; i < maxRobots; i++)
        {
            int baseIdx = i * 4;
            if (robots != null && i < robots.Count && robots[i] != null)
            {
                RobotBattery battery = (_batteries != null && i < _batteries.Length) ? _batteries[i] : null;
                float energyPercent = battery != null ? battery.EnergyPercent : 1f;
                bool busy = (busyFlags != null && i < busyFlags.Length) ? busyFlags[i] : false;

                s[baseIdx + 0] = 0f;
                s[baseIdx + 1] = energyPercent;
                s[baseIdx + 2] = busy ? 1f : 0f;
                s[baseIdx + 3] = energyPercent;
            }
            else
            {
                s[baseIdx + 0] = 1f;
                s[baseIdx + 1] = 0f;
                s[baseIdx + 2] = 1f;
                s[baseIdx + 3] = 0f;
            }
        }

        int ctx = maxRobots * 4;
        s[ctx + 0] = 0f;
        s[ctx + 1] = Mathf.Clamp01(GetIdleCount(busyFlags) / (float)Mathf.Max(1, maxRobots));

        return s;
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    List<int> GetIdleRobotIndices(bool[] busyFlags)
    {
        var list = new List<int>();
        if (robots == null) return list;
        for (int i = 0; i < robots.Count; i++)
        {
            if (robots[i] == null) continue;
            bool busy = (busyFlags != null && i < busyFlags.Length) ? busyFlags[i] : false;
            if (busy) continue;
            RobotBattery battery = (_batteries != null && i < _batteries.Length) ? _batteries[i] : null;
            if (battery != null && !battery.CanDispatch) continue;
            list.Add(i);
        }
        return list;
    }

    int GetIdleCount(bool[] busyFlags) => GetIdleRobotIndices(busyFlags).Count;

    int NearestIdle(List<int> idleIndices, Vector3 targetPos)
    {
        int best = idleIndices[0];
        float bestDist = float.MaxValue;
        foreach (int idx in idleIndices)
        {
            if (robots == null || idx >= robots.Count || robots[idx] == null) continue;
            float d = Vector3.Distance(robots[idx].transform.position, targetPos);
            if (d < bestDist) { bestDist = d; best = idx; }
        }
        return best;
    }

    void HandleReply(int chosenRobotIndex, float[] qValues, float epsilon, bool isRandom)
    {
        if (!_waitingReply) return;
        _repliedRobotIndex = chosenRobotIndex;
        _replyQValues = qValues;
        _replyIsRandom = isRandom;
        _hasReply = true;
        _waitingReply = false;

        if (debugLogs)
            Debug.Log($"[RobotDqnAgent] reply: robot={chosenRobotIndex} random={isRandom} eps(py)={epsilon:F3}");
    }

    public void RecordDirectDispatch(int robotIdx, int nodeId, bool[] busyFlags, Vector3 targetPos, int repairTU, float[] precomputedState = null)
    {
        float[] stateVec = precomputedState ?? BuildStateVector(busyFlags, targetPos, nodeId, repairTU);
        RobotBattery battery = (_batteries != null && robotIdx < _batteries.Length) ? _batteries[robotIdx] : null;
        float dist = (robots != null && robotIdx < robots.Count && robots[robotIdx] != null)
            ? Vector3.Distance(robots[robotIdx].transform.position, targetPos) : 0f;

        _pending[nodeId] = new PendingRobotTransition
        {
            robotIndex = robotIdx,
            nodeId = nodeId,
            stateT = stateVec,
            batteryBeforePct = battery != null ? battery.EnergyPercent : 1f,
            distanceAtDispatch = dist,
            recordTime = Time.realtimeSinceStartup
        };
    }
}
