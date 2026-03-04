using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

/// <summary>
/// Python DQN.py와 TCP로 통신하는 클라이언트.
/// - JSON 한 줄(line) 단위 송수신
/// - transition 전송: {"type":"transition", ...}
/// - q_update 수신:   {"type":"q_update", "node_ids":[...], "q_values":[...]}
/// - action_reply 수신: {"type":"action_reply", ...}  (DqnAgent에서 사용)
/// </summary>
public class DqnTcpClient : MonoBehaviour
{
    [Header("TCP 설정")]
    public string host = "127.0.0.1";
    public int port = 50007;
    public bool autoConnectOnStart = true;

    [Header("Debug 옵션")]
    public bool debugLogs = true;

    TcpClient client;
    NetworkStream stream;
    Thread recvThread;
    volatile bool running = false;
    readonly object sendLock = new object();

        // === (추가) RecvLoop에서 넘어온 로그 메시지 큐 ===
    readonly Queue<string> logQueue = new Queue<string>();
    readonly object logLock = new object();
    // === RecvLoop → 메인 스레드로 넘길 수신 라인 큐 ===
    readonly Queue<string> pendingLines = new Queue<string>();
    readonly object pendingLock = new object();
    // === q_update 큐 (기존) ===
    struct QUpdateItem
    {
        public int nodeId;
        public float qValue;
    }

    readonly Queue<QUpdateItem> qUpdateQueue = new Queue<QUpdateItem>();
    readonly object qLock = new object();

    /// <summary>
    /// 메인 스레드에서 호출되는 콜백 (node_id, q_value)
    /// </summary>
    public Action<int, float> OnQUpdate;

    // === action_reply 큐 (신규) ===
    struct ActionReplyItem
    {
        public int chosenNodeId;
        public int[] candidateNodeIds;
        public float[] qValues;
        public float epsilon;
        public bool isRandom;
    }

    readonly Queue<ActionReplyItem> actionReplyQueue = new Queue<ActionReplyItem>();
    readonly object actionLock = new object();

    /// <summary>
    /// 메인 스레드에서 호출되는 콜백
    /// (chosenNodeId, candidateNodeIds, qValues, epsilon, isRandom)
    /// </summary>
    public Action<int, int[], float[], float, bool> OnActionReply;

    // === 수신 메시지용 DTO ===
    [Serializable]
    class BaseMsg
    {
        public string type;
    }

    [Serializable]
    class QUpdateMsg
    {
        public string type;
        public int[] node_ids;
        public float[] q_values;
    }

    [Serializable]
    class ActionReplyMsg
    {
        public string type;
        public int chosen_node_id;
        public int[] candidate_node_ids;
        public float[] q_values;
        public float epsilon;
        public bool is_random;
    }

    // // === DqnAgent용: 모든 수신 라인 큐 ===
    // readonly Queue<string> lineQueue = new Queue<string>();
    // readonly object lineLock = new object();

    /// <summary>
    /// 현재 TCP 연결 상태 (DqnAgent에서 사용)
    /// </summary>
    public bool IsConnected => client != null && client.Connected;

    void Start()
    {
        if (autoConnectOnStart)
            Connect();
    }

    void Update()
    {
        // -1) (추가) RecvLoop에서 쌓아둔 로그를 메인 스레드에서 출력
        while (true)
        {
            string log;
            lock (logLock)
            {
                if (logQueue.Count == 0)
                    break;
                log = logQueue.Dequeue();
            }

            Debug.Log(log);
        }

        // 0) RecvLoop에서 넘어온 수신 라인들 먼저 처리
        while (true)
        {
            string line;
            lock (pendingLock)
            {
                if (pendingLines.Count == 0)
                    break;
                line = pendingLines.Dequeue();
            }

            // 이제는 메인 스레드에서만 HandleLine을 호출
            HandleLine(line);
        }

        // 1) q_update 처리 (기존)
        if (OnQUpdate != null)
        {
            while (true)
            {
                QUpdateItem item;
                lock (qLock)
                {
                    if (qUpdateQueue.Count == 0)
                        break;
                    item = qUpdateQueue.Dequeue();
                }

                try
                {
                    OnQUpdate?.Invoke(item.nodeId, item.qValue);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[DqnTcpClient] OnQUpdate callback error: {e}");
                }
            }
        }

        // 2) action_reply 처리 (기존)
        if (OnActionReply != null)
        {
            while (true)
            {
                ActionReplyItem item;
                lock (actionLock)
                {
                    if (actionReplyQueue.Count == 0)
                        break;
                    item = actionReplyQueue.Dequeue();
                }

                try
                {
                    OnActionReply?.Invoke(
                        item.chosenNodeId,
                        item.candidateNodeIds,
                        item.qValues,
                        item.epsilon,
                        item.isRandom
                    );
                }
                catch (Exception e)
                {
                    Debug.LogError($"[DqnTcpClient] OnActionReply callback error: {e}");
                }
            }
        }
    }


    void OnDestroy()
    {
        Close();
    }

    public void Connect()
    {
        if (client != null)
            return;

        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();

            running = true;
            recvThread = new Thread(RecvLoop);
            recvThread.IsBackground = true;
            recvThread.Start();

            if (debugLogs)
                Debug.Log($"[DqnTcpClient] Connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] Connect error: {e}");
            Close();
        }
    }

    public void Close()
    {
        running = false;

        try { stream?.Close(); } catch { }
        try { client?.Close(); } catch { }

        stream = null;
        client = null;

        // 대기 중인 ReadLineBlocking 깨우기용
        // lock (lineLock)
        // {
        //     Monitor.PulseAll(lineLock);
        // }
    }

    /// <summary>
    /// JSON 문자열을 한 줄(line)로 전송
    /// </summary>
    public void SendJsonLine(string json)
    {
        if (client == null || stream == null || !client.Connected)
            return;

        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
            lock (sendLock)
            {
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush();
            }

            if (debugLogs)
                Debug.Log($"[DqnTcpClient] Sent: {json}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] Send error: {e}");
            Close();
        }
    }

    // /// <summary>
    // /// RecvLoop에서 잘라낸 "한 줄"을 lineQueue에 넣고, 대기 중인 스레드를 깨움.
    // /// (DqnAgent.ReadLineBlocking에서 사용)
    // /// </summary>
    // void EnqueueLineForBlockingRead(string line)
    // {
    //     lock (lineLock)
    //     {
    //         lineQueue.Enqueue(line);
    //         Monitor.PulseAll(lineLock);
    //     }
    // }

    // /// <summary>
    // /// DqnAgent에서 action_reply 등을 기다릴 때 사용하는 blocking read.
    // /// - timeoutMs 내에 수신된 첫 번째 라인을 반환
    // /// - 타임아웃/연결 끊김 시 null 반환
    // /// </summary>
    // public string ReadLineBlocking(int timeoutMs = 2000)
    // {
    //     if (client == null || stream == null || !client.Connected)
    //         return null;

    //     lock (lineLock)
    //     {
    //         // 이미 큐에 라인이 있으면 바로 반환
    //         if (lineQueue.Count > 0)
    //             return lineQueue.Dequeue();

    //         int remaining = timeoutMs;
    //         DateTime start = DateTime.UtcNow;

    //         while (true)
    //         {
    //             if (client == null || !client.Connected)
    //                 return null;

    //             if (lineQueue.Count > 0)
    //                 return lineQueue.Dequeue();

    //             if (remaining <= 0)
    //                 return null;

    //             Monitor.Wait(lineLock, remaining);

    //             remaining = timeoutMs - (int)(DateTime.UtcNow - start).TotalMilliseconds;
    //         }
    //     }
    // }

    void RecvLoop()
    {
        byte[] buffer = new byte[4096];
        StringBuilder sb = new StringBuilder();

        try
        {
            while (running && client != null && client.Connected)
            {
                int n = stream.Read(buffer, 0, buffer.Length);
                if (n <= 0)
                {
                    if (debugLogs)
                    {
                        lock (logLock)
                        {
                            logQueue.Enqueue("[DqnTcpClient] Connection closed by server.");
                        }
                    }
                    break;
                }

                string chunk = Encoding.UTF8.GetString(buffer, 0, n);
                sb.Append(chunk);

                while (true)
                {
                    string current = sb.ToString();
                    int idx = current.IndexOf('\n');
                    if (idx < 0)
                        break;

                    string line = current.Substring(0, idx).Trim();
                    sb.Remove(0, idx + 1);

                if (!string.IsNullOrEmpty(line))
                {
                    // 1) 메인 스레드에서 처리할 수 있도록 pendingLines에만 쌓기
                    lock (pendingLock)
                    {
                        pendingLines.Enqueue(line);
                    }

                    // 2) (현재 구조에선 ReadLineBlocking을 사용하지 않으므로
                    //     lineQueue에 따로 쌓지 않는다)
                    // EnqueueLineForBlockingRead(line);
                }


                }
            }
        }
        catch (Exception e)
        {
            // 여기서는 직접 Debug.LogError 찍지 말고 큐에 쌓기
            lock (logLock)
            {
                logQueue.Enqueue($"[DqnTcpClient] RecvLoop error: {e}");
            }
        }

        running = false;
    }

    void HandleLine(string line)
    {
        try
        {
            // 먼저 type만 파싱
            var baseMsg = JsonUtility.FromJson<BaseMsg>(line);
            if (baseMsg == null || string.IsNullOrEmpty(baseMsg.type))
            {
                if (debugLogs)
                    Debug.Log($"[DqnTcpClient] recv line (no type): {line}");
                return;
            }

            // --------------------
            // q_update 처리 (기존)
            // --------------------
            if (baseMsg.type == "q_update")
            {
                var msg = JsonUtility.FromJson<QUpdateMsg>(line);
                if (msg != null && msg.node_ids != null && msg.q_values != null)
                {
                    int count = Mathf.Min(msg.node_ids.Length, msg.q_values.Length);
                    for (int i = 0; i < count; i++)
                    {
                        lock (qLock)
                        {
                            qUpdateQueue.Enqueue(new QUpdateItem
                            {
                                nodeId = msg.node_ids[i],
                                qValue = msg.q_values[i]
                            });
                        }
                    }

                    if (debugLogs)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            Debug.Log($"[DqnTcpClient] q_update: node_id={msg.node_ids[i]}, Q={msg.q_values[i]:+0.000}");
                        }
                    }
                }
                else
                {
                    if (debugLogs)
                        Debug.Log($"[DqnTcpClient] q_update parse error or empty arrays: {line}");
                }

                return;
            }

            // --------------------
            // action_reply 처리 (신규)
            // --------------------
            if (baseMsg.type == "action_reply")
            {
                var msg = JsonUtility.FromJson<ActionReplyMsg>(line);
                if (msg != null)
                {
                    var item = new ActionReplyItem
                    {
                        chosenNodeId = msg.chosen_node_id,
                        candidateNodeIds = msg.candidate_node_ids,
                        qValues = msg.q_values,
                        epsilon = msg.epsilon,
                        isRandom = msg.is_random
                    };

                    lock (actionLock)
                    {
                        actionReplyQueue.Enqueue(item);
                    }

                    if (debugLogs)
                    {
                        string candStr = (msg.candidate_node_ids != null)
                            ? string.Join(",", msg.candidate_node_ids)
                            : "null";
                        Debug.Log(
                            $"[DqnTcpClient] action_reply: chosen={msg.chosen_node_id}, " +
                            $"candidates=[{candStr}], eps={msg.epsilon:0.000}, random={msg.is_random}"
                        );
                    }
                }
                else
                {
                    if (debugLogs)
                        Debug.Log($"[DqnTcpClient] action_reply parse error: {line}");
                }

                return;
            }

            // --------------------
            // 그 외 타입
            // --------------------
            if (debugLogs)
                Debug.Log($"[DqnTcpClient] recv line (unknown type={baseMsg.type}): {line}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] JSON parse error: {e}, line={line}");
        }
    }
}