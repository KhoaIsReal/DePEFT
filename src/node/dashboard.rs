/// Embedded HTML/CSS/JavaScript Web Explorer and Management Portal for DePEFT Node.
pub const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DePEFT Network Explorer & Node Portal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #111827;
            --bg-card-hover: #1f2937;
            --border-color: #374151;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background-color: rgba(17, 24, 39, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-logo {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .badge-live::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--accent-green);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--accent-green);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            width: 100%;
            flex: 1;
        }

        /* Metric Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: transform 0.2s, border-color 0.2s;
        }

        .metric-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }

        .metric-title {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #fff;
        }

        /* Tabs & Panels */
        .tabs {
            display: flex;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0.75rem 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }

        .tab-btn.active {
            color: var(--accent-cyan);
            border-bottom-color: var(--accent-cyan);
        }

        .panel {
            display: none;
        }

        .panel.active {
            display: block;
        }

        /* Tables & Lists */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.85rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        tbody tr:hover {
            background-color: var(--bg-card-hover);
        }

        .code-pill {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            color: var(--accent-cyan);
        }

        .tag-pill {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }

        .tag-qlora {
            background: rgba(139, 92, 246, 0.2);
            color: var(--accent-purple);
            border: 1px solid var(--accent-purple);
        }

        /* Forms & Quick Actions */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .form-input {
            background: #1f2937;
            border: 1px solid var(--border-color);
            color: #fff;
            padding: 0.65rem 0.85rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--accent-cyan);
        }

        .btn {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #fff;
            border: none;
            padding: 0.7rem 1.25rem;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .btn:hover {
            opacity: 0.9;
        }

        /* Quick JSON Explorer Modal / Terminal view */
        .terminal-box {
            background: #000;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #10b981;
            max-height: 250px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-logo">🌐 DePEFT Node Explorer</div>
            <div class="badge-live" id="status-live">ONLINE</div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text-muted);">
            Endpoint: <span style="color: var(--accent-cyan);" id="node-endpoint">http://127.0.0.1:8545</span>
        </div>
    </header>

    <div class="container">
        <!-- 4 Top Level Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Block Height</div>
                <div class="metric-value" id="metric-block-height">#0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Active PEFT Tasks</div>
                <div class="metric-value" id="metric-tasks-count">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">IPFS CAS Artifacts</div>
                <div class="metric-value" id="metric-storage-count">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Connected P2P Peers</div>
                <div class="metric-value" id="metric-peers-count">0</div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tasks')">📋 Active Tasks & Tournaments</button>
            <button class="tab-btn" onclick="switchTab('balances')">💰 Accounts & Balances</button>
            <button class="tab-btn" onclick="switchTab('p2p')">🌐 P2P Overlay Swarm</button>
            <button class="tab-btn" onclick="switchTab('tools')">🛠️ Quick API Test Console</button>
        </div>

        <!-- Panel 1: Tasks -->
        <div id="panel-tasks" class="panel active">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Registered Fine-Tuning Tasks</div>
                    <button class="btn" onclick="fetchData()">🔄 Refresh</button>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Task ID</th>
                            <th>Base Model</th>
                            <th>PEFT Method</th>
                            <th>Target Modules</th>
                            <th>Bounty Pool</th>
                            <th>Client Address</th>
                            <th>Deadline Block</th>
                        </tr>
                    </thead>
                    <tbody id="tasks-table-body">
                        <tr>
                            <td colspan="7" style="text-align:center; color: var(--text-muted);">No tasks active on network</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Panel 2: Account Balances & Testnet Faucet -->
        <div id="panel-balances" class="panel">
            <div class="card">
                <div class="card-title" style="margin-bottom: 1rem;">Query Account Balance & Testnet Faucet</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Account ID or Public Key (0x...)</label>
                        <input type="text" id="input-account" class="form-input" placeholder="e.g. client-ai-labs or 0x404b...">
                    </div>
                    <div class="form-group" style="display: flex; flex-direction: row; gap: 0.5rem; align-items: flex-end;">
                        <button class="btn" onclick="queryBalance()">🔍 Check Balance</button>
                        <button class="btn" style="background: linear-gradient(135deg, var(--accent-green), var(--accent-cyan));" onclick="claimFaucet()">🚰 Claim Faucet (10,000 $DEPEFT)</button>
                    </div>
                </div>
                <div id="balance-result" style="margin-top: 1rem; font-size: 1.1rem; font-weight: 600; display: none;">
                    Balance: <span style="color: var(--accent-green);" id="balance-value">0</span> $DEPEFT Tokens
                </div>
            </div>
        </div>

        <!-- Panel 3: P2P Network -->
        <div id="panel-p2p" class="panel">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">P2P Swarm Connected Peers</div>
                    <button class="btn" onclick="fetchPeers()">🔄 Refresh Peers</button>
                </div>
                <ul id="peer-list" style="list-style: none; display: flex; flex-direction: column; gap: 0.5rem;">
                    <li style="color: var(--text-muted);">Scanning for overlay swarm peers...</li>
                </ul>
            </div>
        </div>

        <!-- Panel 4: Quick API Test Console -->
        <div id="panel-tools" class="panel">
            <div class="card">
                <div class="card-title" style="margin-bottom: 1rem;">Raw JSON RPC Console</div>
                <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                    <button class="btn" onclick="runQuickApi('/api/v1/status')">GET /api/v1/status</button>
                    <button class="btn" onclick="runQuickApi('/api/v1/tasks')">GET /api/v1/tasks</button>
                    <button class="btn" onclick="runQuickApi('/api/v1/p2p/peers')">GET /api/v1/p2p/peers</button>
                </div>
                <div class="terminal-box" id="json-console">{ "message": "Click a query button above to inspect JSON response" }</div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('node-endpoint').textContent = window.location.origin;

        function switchTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById('panel-' + tabName).classList.add('active');

            if (tabName === 'tasks') fetchData();
            if (tabName === 'p2p') fetchPeers();
        }

        async function fetchData() {
            try {
                const statusRes = await fetch('/api/v1/status');
                const status = await statusRes.json();
                document.getElementById('metric-block-height').textContent = '#' + status.block_height;
                document.getElementById('metric-tasks-count').textContent = status.tasks_count;
                document.getElementById('metric-storage-count').textContent = status.storage_objects_count;
                document.getElementById('metric-peers-count').textContent = status.connected_peers_count;

                const tasksRes = await fetch('/api/v1/tasks');
                const tasks = await tasksRes.json();
                const tbody = document.getElementById('tasks-table-body');
                
                if (tasks.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">No tasks registered yet</td></tr>';
                } else {
                    tbody.innerHTML = tasks.map(t => {
                        const model = typeof t.base_model_id === 'string' ? t.base_model_id : new TextDecoder().decode(new Uint8Array(t.base_model_id));
                        const modules = t.target_modules ? t.target_modules.map(m => typeof m === 'string' ? m : new TextDecoder().decode(new Uint8Array(m))).join(', ') : 'N/A';
                        return `
                            <tr>
                                <td><span class="code-pill">#${t.task_id}</span></td>
                                <td style="font-weight: 600; color: #fff;">${model}</td>
                                <td><span class="tag-pill tag-qlora">${t.peft_method || 'QLoRA_NF4'}</span></td>
                                <td><span class="code-pill">${modules}</span></td>
                                <td style="color: var(--accent-green); font-weight: 600;">${t.bounty_pool}</td>
                                <td><span class="code-pill">${t.client_address.slice(0, 16)}...</span></td>
                                <td>#${t.epoch_end_block}</td>
                            </tr>
                        `;
                    }).join('');
                }
            } catch (err) {
                console.error("Failed fetching node data", err);
            }
        }

        async function fetchPeers() {
            try {
                const res = await fetch('/api/v1/p2p/peers');
                const data = await res.json();
                const peerList = document.getElementById('peer-list');
                if (!data.connected_peers || data.connected_peers.length === 0) {
                    peerList.innerHTML = '<li style="color: var(--text-muted);">No external peers connected (Running single-node standalone).</li>';
                } else {
                    peerList.innerHTML = data.connected_peers.map(p => `
                        <li style="padding: 0.5rem; background: #1f2937; border-radius: 6px; font-family: 'JetBrains Mono', monospace;">
                            🔗 ${p}
                        </li>
                    `).join('');
                }
            } catch (err) {
                console.error("Failed fetching peers", err);
            }
        }

        async function queryBalance() {
            const acc = document.getElementById('input-account').value.trim();
            if (!acc) return alert("Please enter an Account ID");
            try {
                const res = await fetch('/api/v1/accounts/' + encodeURIComponent(acc) + '/balance');
                const data = await res.json();
                document.getElementById('balance-value').textContent = data.balance.toLocaleString();
                document.getElementById('balance-result').style.display = 'block';
            } catch (err) {
                alert("Failed to query account balance");
            }
        }

        async function claimFaucet() {
            const acc = document.getElementById('input-account').value.trim();
            if (!acc) return alert("Please enter an Account ID");
            try {
                const res = await fetch('/api/v1/faucet', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ account: acc, amount: 10000 })
                });
                const data = await res.json();
                alert("🚰 Faucet Claimed: " + data.message);
                queryBalance();
            } catch (err) {
                alert("Failed to claim from faucet: " + err);
            }
        }

        async function runQuickApi(path) {
            try {
                const res = await fetch(path);
                const data = await res.json();
                document.getElementById('json-console').textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                document.getElementById('json-console').textContent = "Error: " + err;
            }
        }

        // Auto initial load & refresh every 5s
        fetchData();
        setInterval(fetchData, 5000);
    </script>
</body>
</html>
"#;
