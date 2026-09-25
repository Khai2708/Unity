// RobotDqnTcpClient.cs
// Separate TCP client for the Robot-selection DQN (port 50008).
// Mirror of DqnTcpClient but typed for robot_action_reply / robot_transition.

using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

public class RobotDqnTcpClient : MonoBehaviour
{
    [Header("TCP")]
    public string host = "127.0.0.1";
    public int port = 50008;
    public bool autoConnectOnStart = true;

    [Header("Debug")]
    public bool debugLogs = true;

    // ── internal ──────────────────────────────────────────────────────────────
    TcpClient _client;
    NetworkStream _stream;
    Thread _recvThread;
    volatile bool _running = false;
    readonly object _sendLock = new object();

    readonly Queue<string> _logQueue   = new Queue<string>();
    readonly object        _logLock    = new object();

    readonly Queue<string> _pendingLines = new Queue<string>();
    readonly object        _pendingLock  = new object();

    // ── robot_action_reply ────────────────────────────────────────────────────
    public struct RobotActionReplyItem
    {
        public int   chosenRobotIndex;
        public float[] qValues;       // length = maxRobots
        public float epsilon;
        public bool  isRandom;
    }

    readonly Queue<RobotActionReplyItem> _replyQueue = new Queue<RobotActionReplyItem>();
    readonly object                      _replyLock  = new object();

    /// Called on main thread: (chosenRobotIndex, qValues, epsilon, isRandom)
    public Action<int, float[], float, bool> OnRobotActionReply;

    public bool IsConnected => _client != null && _client.Connected;

    // ── DTOs ──────────────────────────────────────────────────────────────────
    [Serializable] class BaseMsg            { public string type; }
    [Serializable] class RobotActionReplyMsg
    {
        public string  type;
        public int     chosen_robot_index;
        public float[] q_values;
        public float   epsilon;
        public bool    is_random;
    }

    // ── lifecycle ─────────────────────────────────────────────────────────────
    void Start()   { if (autoConnectOnStart) Connect(); }
    void OnDestroy() { Close(); }

    void Update()
    {
        // flush log queue
        lock (_logLock)
            while (_logQueue.Count > 0) Debug.Log(_logQueue.Dequeue());

        // dispatch received lines on main thread
        while (true)
        {
            string line;
            lock (_pendingLock)
            {
                if (_pendingLines.Count == 0) break;
                line = _pendingLines.Dequeue();
            }
            HandleLine(line);
        }

        // fire callbacks
        if (OnRobotActionReply != null)
        {
            while (true)
            {
                RobotActionReplyItem item;
                lock (_replyLock)
                {
                    if (_replyQueue.Count == 0) break;
                    item = _replyQueue.Dequeue();
                }
                try { OnRobotActionReply(item.chosenRobotIndex, item.qValues, item.epsilon, item.isRandom); }
                catch (Exception e) { Debug.LogError($"[RobotDqnTcpClient] callback error: {e}"); }
            }
        }
    }

    // ── connect / close ───────────────────────────────────────────────────────
    public void Connect()
    {
        if (_client != null) return;
        try
        {
            _client = new TcpClient();
            _client.Connect(host, port);
            _stream  = _client.GetStream();
            _running = true;
            _recvThread = new Thread(RecvLoop) { IsBackground = true };
            _recvThread.Start();
            if (debugLogs) Debug.Log($"[RobotDqnTcpClient] Connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[RobotDqnTcpClient] Connect error: {e}");
            Close();
        }
    }

    public void Close()
    {
        _running = false;
        try { _stream?.Close(); } catch { }
        try { _client?.Close(); } catch { }
        _stream = null; _client = null;
    }

    // ── send ──────────────────────────────────────────────────────────────────
    public void SendJsonLine(string json)
    {
        if (_client == null || _stream == null || !_client.Connected) return;
        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
            lock (_sendLock) { _stream.Write(bytes, 0, bytes.Length); _stream.Flush(); }
            if (debugLogs) Debug.Log($"[RobotDqnTcpClient] Sent: {json}");
        }
        catch (Exception e) { Debug.LogError($"[RobotDqnTcpClient] Send error: {e}"); Close(); }
    }

    // ── recv thread ───────────────────────────────────────────────────────────
    void RecvLoop()
    {
        byte[]        buffer = new byte[4096];
        StringBuilder sb     = new StringBuilder();
        try
        {
            while (_running && _client != null && _client.Connected)
            {
                int n = _stream.Read(buffer, 0, buffer.Length);
                if (n <= 0) { lock (_logLock) _logQueue.Enqueue("[RobotDqnTcpClient] Server closed."); break; }

                sb.Append(Encoding.UTF8.GetString(buffer, 0, n));
                while (true)
                {
                    string cur = sb.ToString();
                    int idx = cur.IndexOf('\n');
                    if (idx < 0) break;
                    string line = cur.Substring(0, idx).Trim();
                    sb.Remove(0, idx + 1);
                    if (!string.IsNullOrEmpty(line))
                        lock (_pendingLock) _pendingLines.Enqueue(line);
                }
            }
        }
        catch (Exception e) { lock (_logLock) _logQueue.Enqueue($"[RobotDqnTcpClient] RecvLoop: {e}"); }
        _running = false;
    }

    // ── parse ─────────────────────────────────────────────────────────────────
    void HandleLine(string line)
    {
        try
        {
            var baseMsg = JsonUtility.FromJson<BaseMsg>(line);
            if (baseMsg == null || string.IsNullOrEmpty(baseMsg.type)) return;

            if (baseMsg.type == "robot_action_reply")
            {
                var msg = JsonUtility.FromJson<RobotActionReplyMsg>(line);
                if (msg == null) return;
                lock (_replyLock)
                    _replyQueue.Enqueue(new RobotActionReplyItem
                    {
                        chosenRobotIndex = msg.chosen_robot_index,
                        qValues          = msg.q_values,
                        epsilon          = msg.epsilon,
                        isRandom         = msg.is_random
                    });
                if (debugLogs)
                    Debug.Log($"[RobotDqnTcpClient] robot_action_reply: robot={msg.chosen_robot_index} random={msg.is_random}");
                return;
            }

            if (debugLogs) Debug.Log($"[RobotDqnTcpClient] unknown type={baseMsg.type}");
        }
        catch (Exception e) { Debug.LogError($"[RobotDqnTcpClient] parse error: {e}, line={line}"); }
    }
}