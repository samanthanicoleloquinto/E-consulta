<?php
/* ================================================================
   📘 Barangay Forecast Dashboard
   ================================================================ */
$API_BASE = "http://localhost:8000";

function fetch_json($url) {
  $ctx = stream_context_create(['http' => ['timeout' => 10, 'ignore_errors' => true]]);
  $raw = @file_get_contents($url, false, $ctx);
  if ($raw !== false && strlen($raw) > 0) {
    $data = json_decode($raw, true);
    if (json_last_error() === JSON_ERROR_NONE) return $data;
    return ["__error__" => "Invalid JSON from $url: " . json_last_error_msg()];
  }
  return ["__error__" => "Failed to retrieve data from $url"];
}

function show_error_card($title, $msg) {
  echo '<div class="error-box"><strong>' . htmlspecialchars($title) . ':</strong> ' . htmlspecialchars($msg) . '</div>';
}

/* ================================================================
   🔗 API + DB CONNECTIONS
   ================================================================ */
$forecast_url = $API_BASE . "/forecast?with_plot=true";
$metrics  = fetch_json($API_BASE . "/metrics");
$forecast = fetch_json($forecast_url);

$conn = new mysqli("localhost", "root", "", "econsulta");
if ($conn->connect_error) die("Database connection failed: " . $conn->connect_error);

/* ================================================================
   🧮 Get Yearly Data
   ================================================================ */
$csv_data = [];
$res = $conn->query("SELECT year, disease, cases FROM machine_learning ORDER BY year, month_num");
if ($res && $res->num_rows > 0) while ($r = $res->fetch_assoc()) $csv_data[] = $r;

/* ================================================================
   🧱 All Data Tab
   ================================================================ */
$ml_rows = [];
$res2 = $conn->query("SELECT * FROM machine_learning ORDER BY year, month_num");
if ($res2 && $res2->num_rows > 0) while ($r = $res2->fetch_assoc()) $ml_rows[] = $r;
$conn->close();

$selected_year = isset($_GET['year']) ? intval($_GET['year']) : null;
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Barangay Forecast Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
/* ================================================================
   🧭 LAYOUT STRUCTURE (Sidebar + Content)
================================================================ */
body 
{
  margin: 0;
  font-family: "Poppins", "Segoe UI", sans-serif;
  background: #f8fafc;
  color: #1e293b;
}

.main-layout 
{
  display: flex;
  min-height: 100vh;
  background: #f8fafc;
}

/* ================================================================
   🧭 SIDEBAR
   ================================================================ */
.sidebar 
{
  width: 220px;
  background-color: #002c6d;
  padding: 20px 15px;
  color: white;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.sidebar img 
{
  width: 90px;
  margin: 10px auto 20px;
  display: block;
}

.sidebar h2 
{
  font-size: 15px;
  margin-bottom: 20px;
  text-align: center;
  letter-spacing: 0.5px;
  color: #fff;
}

.sidebar a 
{
  display: block;
  color: white;
  text-decoration: none;
  margin: 8px 0;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  width: 100%;
  transition: background 0.2s, padding-left 0.2s;
}

.sidebar a:hover 
{
  background-color: #001c47;
  padding-left: 16px;
}

/* ================================================================
   📘 MAIN CONTENT AREA
   ================================================================ */
.main-content 
{
  flex: 1;
  padding: 30px;
}

/* ================================================================
   📘 HEADER
   ================================================================ */
header 
{
  color: #031f5a82;
  text-align: center;
  padding: 35px 15px;
}

header h1 
{
  margin: 0;
  font-size: 50px;
  font-weight: 700;
}

/* ================================================================
   📑 TAB STYLES
   ================================================================ */
.tabs 
{
  display: flex;
  flex-wrap: wrap;
  border-bottom: 2px solid #cbd5e1;
  margin-bottom: 25px;
  gap: 10px;
}

.tab 
{
  background: #e0e7ff;
  color: #1e3a8a;
  padding: 12px 22px;
  font-weight: 600;
  border-radius: 8px 8px 0 0;
  cursor: pointer;
}

.tab.active 
{
  background: #fff;
  border: 2px solid #2563eb;
  border-bottom: none;
  color: #2563eb;
}

.tab-content 
{
  display: none;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 30px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

.tab-content.active 
{ 
  display: block; 
}

h2 
{
  font-size: 22px;
  font-weight: 700;
  color: #1e3a8a;
  border-left: 4px solid #2563eb;
  padding-left: 10px;
  margin-bottom: 20px;
}

/* ================================================================
   🗓 YEAR SELECT
   ================================================================ */
.year-select 
{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: #fff;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 25px;
}

.year-select label 
{ 
  font-weight: 600; 
  color: #2563eb; 
}

.year-select select 
{
  border: 2px solid #cbd5e1;
  border-radius: 6px;
  padding: 6px 10px;
  background: #f8fafc;
  font-weight: 500;
}

/* ================================================================
   📊 CHART + TABLE LAYOUT
   ================================================================ */
.section-grid 
{
  display: flex;
  flex-wrap: wrap;
  gap: 25px;
}

.graph-container, .table-container 
{
  flex: 1 1 500px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 25px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

.graph-container h3 
{
  color: #1e40af;
  margin-bottom: 10px;
  text-align: center;
}

canvas 
{ width: 100% !important; 
  height: 400px !important; 
}

/* ================================================================
   📈 METRICS
   ================================================================ */
.metrics 
{
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 15px;
  margin-bottom: 25px;
}

.metric 
{
  flex: 1 1 160px;
  text-align: center;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 16px;
  background: #f9fafb;
  box-shadow: 0 3px 8px rgba(0,0,0,0.05);
}

.metric span 
{ 
  font-size: 14px; 
  color: #475569; 
}

.metric strong 
{ 
  display: block; 
  color: #2563eb; 
  font-size: 22px; 
  margin-top: 5px; 
}

/* ================================================================
   📋 TABLES
   ================================================================ */
.table-container table 
{
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.table-container th 
{
  background: #2563eb;
  color: #fff;
  text-align: left;
  padding: 10px;
  position: relative;
}

.table-container td 
{
  padding: 8px 10px;
  border-bottom: 1px solid #e2e8f0;
}

.table-container tr:nth-child(even) 
{ 
  background: #f9fafb;
}

/* ================================================================
   📱 RESPONSIVE
   ================================================================ */
@media (max-width: 900px) 
{
  .main-layout 
  { 
    flex-direction: column; 
  }

  .sidebar 
  { 
    width: 100%; 
    flex-direction: row; 
    flex-wrap: wrap; 
    justify-content: center; 
  }

  .main-content 
  { 
    padding: 15px; 
  }

  .sidebar a 
  { 
    font-size: 12px; 
    padding: 6px 8px; 
  }
}
</style>
</head>

<body>
<div class="main-layout">
  <div class="sidebar">
    <img src="pineda_logo.png" alt="Barangay Logo">
    <h2>Admin Panel</h2>
     <a href="admin_dashboard.php">Dashboard</a>
    <a href="admin_accountapproval.php">Account Approval</a>
    <a href="admin_residents.php">Residents</a>
    <a href="admin_consult.php">Consultation</a>
    <a href="admin_pregnant.php">Pregnant</a>
    <a href="admin_infant.php">Infants</a>
    <a href="admin_familyplan.php">Family Planning</a>
    <a href="admin_view_request.php">Free Medicine</a>
    <a href="admin_add_stock.php">Medicine Stocks</a>
    <a href="admin_tbdots.php">TB DOTS</a>
    <a href="admin_toothextraction.php">Free Tooth Extraction</a>
    <a href="admin_doctors.php">Doctor Account Management</a>
    <a href="admin_logs.php">Logs Details</a>
    <a href="admin_announcements.php">Announcements</a>
    <a href="admin_calendar.php">Schedule Calendar</a>
    <a href="admin_chat.php">Client Messages</a>
    <a href="forecast_results.php">Diseases Forecasting</a>
    <a href="admin_login.php">Logout</a>
  </div>

  <div class="main-content">
    <header>
      <h1>Barangay Forecast Dashboard</h1>
    </header>

    <div class="container">
      <div class="tabs">
        <div class="tab active" onclick="openTab('summary', event)">📊 Summary</div>
        <div class="tab" onclick="openTab('forecast', event)">📈 Forecasting</div>
        <div class="tab" onclick="openTab('alldata', event)">📋 All Data</div>
      </div>

      <!-- ================================================================
           📊 SUMMARY TAB
           ================================================================ -->
      <div id="summary" class="tab-content active">
        <h2>Disease Summary by Year</h2>
        <?php
        $yearly_summary = [];
        foreach ($csv_data as $row) {
          $y = intval($row['year']); $d = $row['disease']; $c = intval($row['cases']);
          $yearly_summary[$y][$d] = ($yearly_summary[$y][$d] ?? 0) + $c;
        }
        $years = array_keys($yearly_summary); sort($years);
        $default_year = $selected_year ?? end($years);
        ?>
        <form method="GET" class="year-select">
          <label for="year">Select Year:</label>
          <select name="year" id="year" onchange="this.form.submit()">
            <?php foreach ($years as $y): ?>
              <option value="<?= $y ?>" <?= $y==$default_year?'selected':'' ?>><?= $y ?></option>
            <?php endforeach; ?>
          </select>
        </form>

        <div class="section-grid">
          <div class="graph-container">
            <h3>Actual Cases — <?= $default_year ?></h3>
            <canvas id="chart_summary"></canvas>
          </div>
          <div class="table-container">
            <table>
              <thead><tr><th>Disease</th><th>Total Cases</th></tr></thead>
              <tbody>
                <?php foreach ($yearly_summary[$default_year] as $d => $c): ?>
                  <tr><td><?= htmlspecialchars($d) ?></td><td><?= htmlspecialchars($c) ?></td></tr>
                <?php endforeach; ?>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- ================================================================
           📈 FORECAST TAB
           ================================================================ -->
      <div id="forecast" class="tab-content">
        <h2>Forecast and Model Accuracy</h2>
        <div class="metrics">
          <div class="metric"><span>R² Accuracy</span><strong><?= number_format($metrics['r2_accuracy'] ?? 0, 2) ?>%</strong></div>
          <div class="metric"><span>MAE</span><strong><?= number_format($metrics['mae'] ?? 0, 2) ?></strong></div>
          <div class="metric"><span>RMSE</span><strong><?= number_format($metrics['rmse'] ?? 0, 2) ?></strong></div>
        </div>

        <div class="section-grid">
          <div class="graph-container">
            <h3>Predicted Disease Cases</h3>
            <canvas id="chart_forecast"></canvas>
          </div>
          <div class="table-container">
  <table>
    <thead>
      <tr><th>Year</th><th>Month</th><th>Predicted Cases</th></tr>
    </thead>
    <tbody>
      <?php if (!empty($forecast['forecast'])):
        foreach ($forecast['forecast'] as $r):
          $year = htmlspecialchars($r['year'] ?? '—'); // 🆕 Added year
          $month = htmlspecialchars($r['month'] ?? '—');
          $total = intval($r['predicted_cases'] ?? 0);
          echo "<tr><td>$year</td><td>$month</td><td>$total</td></tr>";
        endforeach;
      endif; ?>
    </tbody>
  </table>
</div>

        </div>
      </div>

      <!-- ================================================================
           📋 ALL DATA TAB
           ================================================================ -->
      <div id="alldata" class="tab-content">
        <h2>Machine Learning Data (SQL Table)</h2>
        <div class="table-container" style="overflow-x:auto; max-height:600px;">
          <table>
            <thead>
              <tr>
                <?php foreach(array_keys($ml_rows[0]??[]) as $head) echo "<th>".strtoupper(htmlspecialchars($head))."</th>"; ?>
              </tr>
            </thead>
            <tbody>
              <?php foreach($ml_rows as $r){ echo "<tr>"; foreach($r as $c){ echo "<td>".htmlspecialchars($c)."</td>"; } echo "</tr>"; } ?>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     📜 JAVASCRIPT
     ================================================================ -->
<script>
function openTab(id, e){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  e.target.classList.add('active');
}

/* ======== SUMMARY CHART ======== */
const summaryLabels = <?= json_encode(array_keys($yearly_summary[$default_year])) ?>;
const summaryData   = <?= json_encode(array_values($yearly_summary[$default_year])) ?>;
new Chart(document.getElementById('chart_summary'), {
  type: 'bar',
  data: { labels: summaryLabels, datasets: [{
    label: 'Cases',
    data: summaryData,
    backgroundColor: (ctx)=> {
      const g = ctx.chart.ctx.createLinearGradient(0,0,0,400);
      g.addColorStop(0,'#60a5fa'); g.addColorStop(1,'#2563eb'); return g;
    },
    borderWidth: 1
  }]},
  options: {
    responsive:true,
    plugins:{ 
      legend:{display:false},
      title:{ display:true, text:'Total Reported Cases', color:'#1e293b', font:{size:16,weight:'bold'} }
    },
    scales:{ y:{beginAtZero:true,title:{display:true,text:'Cases'}}, x:{ticks:{color:'#1e293b'}} }
  }
});

/* ======== FORECAST CHART ======== */
<?php
$forecast_months = [];
$forecast_cases = [];
if (!empty($forecast['forecast'])) {
  foreach ($forecast['forecast'] as $f) {
    $forecast_months[] = $f['month'];
    $forecast_cases[]  = intval($f['predicted_cases']);
  }
}
?>
new Chart(document.getElementById('chart_forecast'), {
  type: 'bar',
  data: { labels: <?= json_encode($forecast_months) ?>, datasets: [{
    label: 'Predicted Cases',
    data: <?= json_encode($forecast_cases) ?>,
    backgroundColor:(ctx)=>{
      const g=ctx.chart.ctx.createLinearGradient(0,0,0,400);
      g.addColorStop(0,'#93c5fd'); g.addColorStop(1,'#2563eb'); return g;
    },
    borderWidth:1
  }]},
  options:{
    responsive:true,
    plugins:{
      legend:{display:false},
      title:{ display:true, text:'Forecasted Disease Cases', color:'#1e293b', font:{size:16,weight:'bold'} }
    },
    scales:{ y:{beginAtZero:true,title:{display:true,text:'Predicted Cases'}}, x:{ticks:{color:'#1e293b'}} }
  }
});
</script>
</body>
</html>
