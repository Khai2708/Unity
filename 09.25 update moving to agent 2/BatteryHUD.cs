using UnityEngine;
using System.Linq;

/// <summary>
/// Runtime on-screen overlay showing every robot's battery level.
/// Attach to any single GameObject in the scene (e.g. an empty "HUD" object).
/// Automatically finds all RobotBattery components -- no wiring needed.
/// </summary>
public class BatteryHUD : MonoBehaviour
{
    [Header("Display")]
    public bool showHUD = true;
    public Vector2 screenOffset = new Vector2(10, 10);
    public float lineHeight = 22f;
    public float barWidth = 160f;

    [Header("Refresh")]
    [Tooltip("How often (seconds) to re-scan the scene for robots. Doesn't need to be fast.")]
    public float refreshInterval = 1f;

    private RobotBattery[] batteries;
    private float nextRefresh = 0f;
    private Texture2D fillTex;

    void Awake()
    {
        fillTex = Texture2D.whiteTexture;
    }

    void Update()
    {
        if (Time.time >= nextRefresh)
        {
            batteries = FindObjectsOfType<RobotBattery>().OrderBy(b => b.name).ToArray();
            nextRefresh = Time.time + refreshInterval;
        }
    }

    void OnGUI()
    {
        if (!showHUD || batteries == null || batteries.Length == 0) return;

        var labelStyle = new GUIStyle(GUI.skin.label) { fontSize = 14, normal = { textColor = Color.white } };
        var titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 14, fontStyle = FontStyle.Bold, normal = { textColor = Color.white } };

        float boxWidth = barWidth + 180f;
        float boxHeight = batteries.Length * (lineHeight + 8f) + 30f;

        GUI.Box(new Rect(screenOffset.x - 5, screenOffset.y - 5, boxWidth, boxHeight), string.Empty);

        float y = screenOffset.y;
        GUI.Label(new Rect(screenOffset.x, y, boxWidth - 10, lineHeight), "Robot Battery", titleStyle);
        y += lineHeight + 6f;

        foreach (var b in batteries)
        {
            if (b == null) continue;

            float pct = b.EnergyPercent;
            string status = b.unlimitedEnergy
                ? "unlimited"
                : $"{b.CurrentEnergy:F0}/{b.maxEnergy:F0} ({pct:P0})" +
                  (b.IsCharging ? " [charging]" : "") +
                  (b.IsLow ? " [LOW]" : "");

            GUI.Label(new Rect(screenOffset.x, y, boxWidth - 10, lineHeight), $"{b.name}: {status}", labelStyle);

            if (!b.unlimitedEnergy)
            {
                Rect barBg = new Rect(screenOffset.x, y + 17, barWidth, 6);
                var prevColor = GUI.color;

                GUI.color = Color.gray;
                GUI.DrawTexture(barBg, fillTex);

                GUI.color = pct < 0.15f ? Color.red : (pct < 0.4f ? Color.yellow : Color.green);
                GUI.DrawTexture(new Rect(barBg.x, barBg.y, barWidth * Mathf.Clamp01(pct), barBg.height), fillTex);

                GUI.color = prevColor;
            }

            y += lineHeight + 8f;
        }
    }
}
