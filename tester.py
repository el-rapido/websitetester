import requests
import time
import sqlite3
import psutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from fpdf import FPDF
from datetime import datetime, timedelta
import threading
import seaborn as sns

# ====== Paths ======
DB_PATH = "test_results.db"
PDF_PATH = "test_report.pdf"
CSV_PATH = "test_results.csv"
SUMMARY_CSV = "summary_by_test_type.csv"
ENHANCED_SUMMARY_CSV = "enhanced_summary.csv"
ENHANCED_PDF_PATH = "enhanced_test_report.pdf"

# ====== Database Setup ======
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (timestamp TEXT, name TEXT, url TEXT, test_type TEXT,
                  status_code INTEGER, response_time REAL, cpu REAL, ram REAL)''')
    conn.commit()
    conn.close()

# ====== Result Logging ======
def log_result(name, url, test_type, status, duration, cpu, ram):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), name, url, test_type, status, duration, cpu, ram))
    conn.commit()
    conn.close()

# ====== Single Request Execution ======
def run_test(target, test_type="General"):
    name, url = target["name"], target["url"]
    try:
        start = time.time()
        response = requests.get(url, timeout=10)
        duration = time.time() - start
        status_code = response.status_code
    except:
        duration = -1
        status_code = 0
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    log_result(name, url, test_type, status_code, duration, cpu, ram)

# ====== Test Implementations ======
def load_test(target, steps=[1, 5, 10]):
    for users in steps:
        print(f"🚀 Load Test: {users} concurrent requests")
        threads = []
        for _ in range(users):
            t = threading.Thread(target=run_test, args=(target, "Load Test"))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        time.sleep(2)

def failure_injection_test(target):
    bad_urls = [
        target["url"] + "/not-found",
        target["url"] + "/timeout",
        "http://invalid.local"
    ]
    for url in bad_urls:
        print(f"🔧 Injecting failure to {url}")
        run_test({"name": "Failure Injection", "url": url}, "Failure Injection")

def rate_limit_test(target, burst=20):
    print(f"⚡ Rate-Limit Test: {burst} quick requests")
    for _ in range(burst):
        run_test(target, "Rate-Limit Test")

def long_stability_test(target, duration=120, interval=10):
    print("📈 Starting Long-Run Stability Test")
    start = time.time()
    while time.time() - start < duration:
        run_test(target, "Long-Run Stability Test")
        time.sleep(interval)

# ====== CSV Export ======
def export_to_csv():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    df.to_csv(CSV_PATH, index=False)
    print(f"📄 CSV report saved to: {CSV_PATH}")

# ====== Basic Graph Generation ======
def generate_basic_report():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM results")
    rows = c.fetchall()
    conn.close()

    if not rows:
        print("⚠️ No data to report.")
        return

    timestamps = [datetime.fromisoformat(r[0]) for r in rows]
    response_times = [r[5] for r in rows]
    cpu_usages = [r[6] for r in rows]
    ram_usages = [r[7] for r in rows]

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, response_times, marker='o')
    plt.title('Response Time Over Time')
    plt.ylabel('Seconds')

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, cpu_usages, color='orange', marker='x')
    plt.title('CPU Usage Over Time')
    plt.ylabel('CPU %')

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, ram_usages, color='green', marker='s')
    plt.title('RAM Usage Over Time')
    plt.ylabel('RAM %')
    plt.xlabel('Timestamp')

    plt.tight_layout()
    plt.savefig('report_graphs.png')
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Test Report", ln=True, align='C')
    pdf.image("report_graphs.png", x=10, y=20, w=190)
    pdf.output(PDF_PATH)
    print(f"📊 PDF report saved to: {PDF_PATH}")

# ====== Basic Summary Analysis ======
def basic_summarize_results():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ CSV file not found. Skipping summary.")
        return

    if df.empty:
        print("⚠️ No data to summarize.")
        return

    print("\n📊 Summary by Test Type:\n")
    summary = (
        df.groupby("test_type")
        .agg(
            total_requests=("status_code", "count"),
            avg_response_time=("response_time", "mean"),
            failures=("status_code", lambda x: sum(x != 200)),
            max_cpu=("cpu", "max"),
            max_ram=("ram", "max")
        )
    )
    summary["success_rate_%"] = 100 * (1 - summary["failures"] / summary["total_requests"])
    summary = summary.round(2)
    print(summary.reset_index().to_string(index=False))

    summary.to_csv(SUMMARY_CSV)
    print(f"\n📄 Summary saved to: {SUMMARY_CSV}")

# ====== Enhanced Graph Generation ======
def generate_enhanced_graphs(db_path=DB_PATH, output_path="enhanced_graphs.png"):
    """Generate enhanced graphs with better visualization and insights"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    
    if df.empty:
        print("⚠️ No data to visualize.")
        return False
    
    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter out failed requests for response time visualization
    df_success = df[df['response_time'] >= 0].copy()
    
    # Create a color map for test types
    test_types = df['test_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_types)))
    color_map = dict(zip(test_types, colors))
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Test Results', fontsize=16)
    
    # 1. Response Time Plot (top left)
    ax1 = axs[0, 0]
    for test_type in test_types:
        mask = df_success['test_type'] == test_type
        if mask.any():
            ax1.scatter(df_success[mask]['timestamp'], df_success[mask]['response_time'], 
                       label=test_type, alpha=0.7, c=[color_map[test_type]], marker='o')
    
    ax1.set_title('Response Time by Test Type')
    ax1.set_ylabel('Response Time (seconds)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Success Rate Plot (top right)
    ax2 = axs[0, 1]
    success_data = []
    
    for test_type in test_types:
        test_df = df[df['test_type'] == test_type]
        success_rate = (test_df['status_code'] == 200).mean() * 100
        success_data.append({'Test Type': test_type, 'Success Rate (%)': success_rate})
    
    success_df = pd.DataFrame(success_data)
    bars = ax2.bar(success_df['Test Type'], success_df['Success Rate (%)'], 
                  color=[color_map[t] for t in success_df['Test Type']])
    
    ax2.set_title('Success Rate by Test Type')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 105)
    ax2.set_xticklabels(success_df['Test Type'], rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. Resource Usage Over Time (bottom left)
    ax3 = axs[1, 0]
    
    # Plot both CPU and RAM on the same subplot
    l1 = ax3.plot(df['timestamp'], df['cpu'], 'r-', alpha=0.7, label='CPU Usage')
    ax3.set_ylabel('CPU Usage (%)', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.set_ylim(0, max(100, df['cpu'].max() * 1.1))
    
    ax3_twin = ax3.twinx()
    l2 = ax3_twin.plot(df['timestamp'], df['ram'], 'b-', alpha=0.7, label='RAM Usage')
    ax3_twin.set_ylabel('RAM Usage (%)', color='b')
    ax3_twin.tick_params(axis='y', labelcolor='b')
    ax3_twin.set_ylim(0, max(100, df['ram'].max() * 1.1))
    
    # Add vertical lines to mark test type transitions
    test_changes = df.loc[df['test_type'] != df['test_type'].shift(1)]
    for idx, row in test_changes.iterrows():
        if idx > 0:  # Skip the first entry which is the start of the first test
            ax3.axvline(x=row['timestamp'], color='k', linestyle='--', alpha=0.5)
    
    ax3.set_title('Resource Usage Over Time')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.tick_params(axis='x', rotation=45)
    
    # Combine legends from both y-axes
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. Response Time Distribution (bottom right)
    ax4 = axs[1, 1]
    
    for test_type in test_types:
        test_data = df_success[df_success['test_type'] == test_type]['response_time']
        if not test_data.empty:
            sns.kdeplot(test_data, ax=ax4, label=test_type, fill=True, alpha=0.3)
    
    ax4.set_title('Response Time Distribution')
    ax4.set_xlabel('Response Time (seconds)')
    ax4.set_ylabel('Density')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    # Create common legend for test types
    handles = [Patch(facecolor=color_map[test_type], label=test_type) for test_type in test_types]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=len(test_types))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Enhanced graphs saved to: {output_path}")
    return True

# ====== Enhanced Summary Analysis ======
def enhanced_summary_analysis(csv_path=CSV_PATH, summary_csv=ENHANCED_SUMMARY_CSV):
    """Generate enhanced summary statistics for all test types"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ CSV file not found. Skipping summary.")
        return None
    
    if df.empty:
        print("⚠️ No data to summarize.")
        return None
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter out failed requests for response time analysis
    df_success = df[df['response_time'] >= 0].copy()
    
    # Calculate test durations
    test_durations = {}
    for test_type in df['test_type'].unique():
        test_df = df[df['test_type'] == test_type]
        if not test_df.empty:
            start_time = test_df['timestamp'].min()
            end_time = test_df['timestamp'].max()
            duration_sec = (end_time - start_time).total_seconds()
            test_durations[test_type] = duration_sec
    
    # Create summary with advanced metrics
    summary = []
    
    for test_type in df['test_type'].unique():
        test_df = df[df['test_type'] == test_type]
        success_df = df_success[df_success['test_type'] == test_type]
        
        # Calculate response time percentiles
        if not success_df.empty:
            p50 = success_df['response_time'].quantile(0.5)
            p90 = success_df['response_time'].quantile(0.9)
            p99 = success_df['response_time'].quantile(0.99)
            std_dev = success_df['response_time'].std()
        else:
            p50 = p90 = p99 = std_dev = float('nan')
        
        # Calculate error rates and resource usage
        total_requests = len(test_df)
        success_count = (test_df['status_code'] == 200).sum()
        error_count = total_requests - success_count
        success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0
        
        # Resource usage patterns
        avg_cpu = test_df['cpu'].mean()
        avg_ram = test_df['ram'].mean()
        peak_cpu = test_df['cpu'].max()
        peak_ram = test_df['ram'].max()
        
        # Add row to summary
        summary.append({
            'Test Type': test_type,
            'Total Requests': total_requests,
            'Success Count': success_count,
            'Error Count': error_count,
            'Success Rate (%)': success_rate,
            'Avg Response Time (s)': success_df['response_time'].mean() if not success_df.empty else float('nan'),
            'Median Response Time (s)': p50,
            'p90 Response Time (s)': p90,
            'p99 Response Time (s)': p99,
            'Std Dev Response Time': std_dev,
            'Min Response Time (s)': success_df['response_time'].min() if not success_df.empty else float('nan'),
            'Max Response Time (s)': success_df['response_time'].max() if not success_df.empty else float('nan'),
            'Test Duration (s)': test_durations.get(test_type, 0),
            'Avg CPU (%)': avg_cpu,
            'Peak CPU (%)': peak_cpu,
            'Avg RAM (%)': avg_ram,
            'Peak RAM (%)': peak_ram
        })
    
    # Convert to DataFrame and export
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.round(2)
    summary_df.to_csv(summary_csv, index=False)
    
    # Print a more readable version of the summary
    print("\n📊 Enhanced Summary by Test Type:\n")
    
    # Display top-level metrics
    display_cols = ['Test Type', 'Total Requests', 'Success Rate (%)', 
                   'Avg Response Time (s)', 'p90 Response Time (s)', 'Test Duration (s)']
    print(summary_df[display_cols].to_string(index=False))
    
    print(f"\n📄 Enhanced summary saved to: {summary_csv}")
    return summary_df

# ====== Enhanced PDF Report Generation ======
def generate_enhanced_report(db_path=DB_PATH, csv_path=CSV_PATH, pdf_path=ENHANCED_PDF_PATH):
    """Generate an enhanced PDF report with graphs and detailed analysis"""
    # Generate enhanced graphs
    graph_generated = generate_enhanced_graphs(db_path, "enhanced_graphs.png")
    
    # Generate enhanced summary
    summary_df = enhanced_summary_analysis(csv_path)
    
    if not graph_generated or summary_df is None or summary_df.empty:
        print("⚠️ Not enough data to generate a complete report.")
        return
    
    # Create PDF with FPDF
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Performance Test Report', 0, 1, 'C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    # Use utf8 to support unicode characters
    pdf.add_page()
    pdf.set_font("Arial", 'B', 15)
    
    
    # Add timestamp
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Add graphs
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Performance Test Visualizations", 0, 1)
    pdf.image("enhanced_graphs.png", x=10, y=None, w=190)
    pdf.ln(5)
    
    # Add summary table
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Test Summary by Test Type", 0, 1)
    
    # Table headers
    pdf.set_font("Arial", 'B', 9)
    headers = ['Test Type', 'Requests', 'Success %', 'Avg Time (s)', 'p90 (s)', 'Duration (s)']
    col_widths = [40, 20, 20, 30, 30, 30]
    
    # Print headers
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()
    
    # Print data rows
    pdf.set_font("Arial", '', 9)
    for _, row in summary_df.iterrows():
        pdf.cell(col_widths[0], 10, str(row['Test Type']), 1)
        pdf.cell(col_widths[1], 10, str(int(row['Total Requests'])), 1)
        pdf.cell(col_widths[2], 10, f"{row['Success Rate (%)']:.1f}%", 1)
        pdf.cell(col_widths[3], 10, f"{row['Avg Response Time (s)']:.3f}", 1)
        pdf.cell(col_widths[4], 10, f"{row['p90 Response Time (s)']:.3f}", 1)
        pdf.cell(col_widths[5], 10, f"{row['Test Duration (s)']:.1f}", 1)
        pdf.ln()
    
    # Add analysis section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Performance Analysis", 0, 1)
    
    # Calculate overall metrics
    success_rate = summary_df['Success Rate (%)'].mean()
    avg_response = summary_df['Avg Response Time (s)'].mean()
    
    # Add analysis text
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 10, f"""
Overall Analysis:
- Average Success Rate: {success_rate:.1f}%
- Average Response Time: {avg_response:.3f} seconds
- Peak CPU Usage: {summary_df['Peak CPU (%)'].max():.1f}%
- Peak RAM Usage: {summary_df['Peak RAM (%)'].max():.1f}%
    """)
    
    # Add test-specific analysis
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Test-Specific Insights:", 0, 1)
    
    for _, row in summary_df.iterrows():
        test_type = row['Test Type']
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 10, f"{test_type}:", 0, 1)
        
        pdf.set_font("Arial", '', 10)
        
        # Generate insights based on test type
        insights = []
        
        if row['Success Rate (%)'] < 90:
            insights.append(f"! Low success rate ({row['Success Rate (%)']:.1f}%) may indicate reliability issues.")
        
        if test_type == "Load Test":
            if row['p99 Response Time (s)'] > 3 * row['Avg Response Time (s)']:
                insights.append("! High response time variance under load.")
            else:
                insights.append("+ System handles load consistently.")
        
        elif test_type == "Rate-Limit Test":
            if row['Error Count'] > 0:
                insights.append("! Rate-limiting may be impacting success rate.")
            else:
                insights.append("+ System handles rapid requests well.")
        
        elif test_type == "Failure Injection":
            if row['Success Rate (%)'] > 50:
                insights.append("! System not properly detecting injected failures.")
            else:
                insights.append("+ System correctly detected injected failures.")
        
        elif test_type == "Long-Run Stability Test":
            if row['Std Dev Response Time'] < 0.5 * row['Avg Response Time (s)']:
                insights.append("+ System shows stable performance over time.")
            else:
                insights.append("! Inconsistent performance over time.")
        
        # Add resource usage insight
        if row['Peak CPU (%)'] > 80:
            insights.append(f"! High CPU usage peak ({row['Peak CPU (%)']:.1f}%).")
        
        # If no insights were generated, add a generic one
        if not insights:
            insights.append("Test completed with no significant issues detected.")
        
        # Add insights to PDF
        for insight in insights:
            pdf.multi_cell(0, 6, f"- {insight}")
        
        pdf.ln(5)
    
    # Save PDF
    pdf.output(pdf_path)
    print(f"📊 Enhanced PDF report saved to: {pdf_path}")

# ====== Main CLI ======
def main():
    init_db()
    user_input = input("Enter target URLs (comma-separated):\n")
    targets = [{"name": url.strip(), "url": url.strip()} for url in user_input.split(",") if url.strip()]
    if not targets:
        print("❌ No valid URLs entered.")
        return

    print("\nSelect tests to run:")
    print("1 - Load Test")
    print("2 - Failure Injection")
    print("3 - Rate-Limit Test")
    print("4 - Long-Run Stability Test")
    print("5 - All Tests")
    choice = input("Enter choice (e.g., 1,3,4): ").split(",")

    for target in targets:
        print(f"\n📍 Running tests on {target['url']}")
        if "5" in choice or "1" in choice:
            load_test(target)
        if "5" in choice or "2" in choice:
            failure_injection_test(target)
        if "5" in choice or "3" in choice:
            rate_limit_test(target)
        if "5" in choice or "4" in choice:
            long_stability_test(target)

    export_to_csv()
    
    # Generate both basic and enhanced reports
    print("\n🔍 Generating basic reports...")
    generate_basic_report()
    basic_summarize_results()
    
    print("\n🔍 Generating enhanced reports...")
    generate_enhanced_report()

    print(f"\n🎉 Done! Results stored in:")
    print(f"- {DB_PATH} (SQLite database)")
    print(f"- {CSV_PATH} (Raw test results)")
    print(f"- {PDF_PATH} (Basic report)")
    print(f"- {SUMMARY_CSV} (Basic summary)")
    print(f"- {ENHANCED_PDF_PATH} (Enhanced report)")
    print(f"- {ENHANCED_SUMMARY_CSV} (Enhanced summary)")

if __name__ == "__main__":
    main()