using System.Collections.Generic;
using UnityEngine;

public class RobotDecisionController : MonoBehaviour
{
    public AStarAgent agent;
    public DqnAgent dqnAgent;          // add reference
    public float decisionInterval = 1.0f;
    private float nextDecisionTime = 0f;
    private bool isWaitingForAction = false;

    void Update()
    {
        if (agent == null || dqnAgent == null) return;
        if (Time.time < nextDecisionTime || isWaitingForAction) return;

        // Get list of candidate node IDs (e.g., all tunnel nodes, or only faulty ones)
        List<int> candidates = GetCandidateNodeIds(); // you need to implement this

        if (candidates.Count == 0) return;

        isWaitingForAction = true;
        StartCoroutine(dqnAgent.CoRequestActionAndPickNode(candidates, 0.1f, 
            (chosenNodeId, isRandom) =>
            {
                // This callback runs when action is chosen
                Debug.Log($"Robot decided to repair node {chosenNodeId}");
                // Set robot target
                var targetTunnel = GetTunnelById(chosenNodeId); // you need a mapping
                if (targetTunnel != null)
                {
                    agent.SetTarget(targetTunnel.transform, true);
                }
                isWaitingForAction = false;
                nextDecisionTime = Time.time + decisionInterval;
            }));
    }

    private List<int> GetCandidateNodeIds()
    {
        // Return all tunnel node IDs (or only faulty ones if you want)
        // You can get them from FactoryEnvManager.Nodes
        List<int> ids = new List<int>();
        foreach (var kv in FactoryEnvManager.Instance.Nodes)
        {
            if (!kv.Value.isSpawner)
                ids.Add(kv.Key);
        }
        return ids;
    }

    private TunnelController GetTunnelById(int id)
    {
        if (FactoryEnvManager.Instance.Nodes.TryGetValue(id, out var data))
            return data.tunnel;
        return null;
    }
}