#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(TunnelController))]
public class TunnelControllerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        
        TunnelController script = (TunnelController)target;
        
        if (script.slowUpstreamWhenFault && script.upstreamSpawners != null && script.upstreamSpawners.Length == 0)
        {
            EditorGUILayout.HelpBox("No upstream spawners assigned. The tunnel won't affect any spawner speeds.", MessageType.Warning);
            
            if (GUILayout.Button("Find and Assign Nearby Spawners"))
            {
                // Try to find spawners in the scene
                var allSpawners = FindObjectsOfType<ProductSpawner>();
                script.upstreamSpawners = new ProductSpawner[0];
                
                // Simple heuristic: find spawners within a certain distance
                foreach (var spawner in allSpawners)
                {
                    float distance = Vector3.Distance(script.transform.position, spawner.transform.position);
                    if (distance < 10f) // Adjust this threshold as needed
                    {
                        ArrayUtility.Add(ref script.upstreamSpawners, spawner);
                    }
                }
                
                EditorUtility.SetDirty(script);
                Debug.Log($"Found {script.upstreamSpawners.Length} nearby spawners for {script.name}");
            }
        }
        
        // Show current speed multiplier
        if (script.slowUpstreamWhenFault)
        {
            EditorGUILayout.LabelField($"Fault Speed Multiplier: {script.faultSpeedMultiplier}x", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox($"When this tunnel fails, upstream spawners will run at {script.faultSpeedMultiplier * 100}% speed.", MessageType.Info);
        }
    }
}
#endif