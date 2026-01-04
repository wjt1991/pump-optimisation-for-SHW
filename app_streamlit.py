"""
Pump Scheduling Optimizer - Streamlit Web Application
Economic optimization based on electricity price forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import pulp
import os
import tempfile
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import io
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo

# Australia/Sydney timezone (AEST/AEDT)
SYDNEY_TZ = ZoneInfo("Australia/Sydney")

def get_sydney_now():
    """Get current time in Sydney timezone"""
    return datetime.now(SYDNEY_TZ)

# Page config
st.set_page_config(
    page_title="Pump Scheduling Optimizer",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============== Data Fetching Functions ==============

def fetch_aemo_predispatch(log_callback=None):
    """Fetch latest PreDispatch data from AEMO NEMWEB"""
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    try:
        url = "https://nemweb.com.au/Reports/Current/PredispatchIS_Reports/"
        log("Accessing AEMO NEMWEB...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        zip_links = []
        for link in links:
            href = link['href']
            if href.endswith('.zip') and 'PREDISPATCHIS' in href.upper():
                zip_links.append(href)
        
        if not zip_links:
            raise Exception("No PreDispatchIS ZIP files found")
        
        log(f"Found {len(zip_links)} files")
        
        # Get latest file
        latest_zip = zip_links[-1]
        if latest_zip.startswith('/'):
            zip_url = 'https://nemweb.com.au' + latest_zip
        else:
            zip_url = url + latest_zip
        
        log(f"Downloading: {latest_zip.split('/')[-1]}")
        
        zip_response = requests.get(zip_url, timeout=60)
        zip_response.raise_for_status()
        
        # Parse CSV
        zip_buffer = io.BytesIO(zip_response.content)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            csv_files = [name for name in zip_file.namelist() if name.endswith('.CSV')]
            if not csv_files:
                raise Exception("No CSV file found in ZIP")
            
            with zip_file.open(csv_files[0]) as csv_file:
                csv_content = csv_file.read().decode('utf-8')
        
        # Parse NEM Format
        lines = csv_content.strip().split('\n')
        
        header_line = None
        header_idx = -1
        table_prefix = None
        
        for idx, line in enumerate(lines):
            if 'REGION_PRICES' in line and line.startswith('I,'):
                header_line = line
                header_idx = idx
                parts = line.split(',')
                if len(parts) >= 2:
                    table_prefix = parts[1]
                break
        
        if header_line is None:
            raise Exception("REGION_PRICES header not found")
        
        columns = [col.strip() for col in header_line.split(',')]
        
        regionid_idx = columns.index('REGIONID')
        datetime_idx = columns.index('DATETIME')
        rrp_idx = columns.index('RRP')
        
        data_rows = []
        data_line_prefix = f'D,{table_prefix},REGION_PRICES'
        
        for line in lines[header_idx + 1:]:
            if line.startswith(data_line_prefix):
                parts = [part.strip() for part in line.split(',')]
                
                if len(parts) > regionid_idx and parts[regionid_idx] == 'NSW1':
                    try:
                        datetime_str = parts[datetime_idx].strip('"')
                        price_str = parts[rrp_idx].strip('"')
                        
                        dt = pd.to_datetime(datetime_str, format='%Y/%m/%d %H:%M:%S')
                        price = float(price_str)
                        
                        data_rows.append({'DateTime': dt, 'Price': price})
                    except:
                        continue
        
        if not data_rows:
            raise Exception("No NSW1 region data found")
        
        df = pd.DataFrame(data_rows)
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        log(f"Successfully fetched {len(df)} records")
        return df
        
    except Exception as e:
        log(f"Error: {e}")
        return None


def parse_predispatch_filename(filename):
    """Parse timestamp from PredispatchIS filename
    Format: PUBLIC_PREDISPATCHIS_YYYYMMDDHHMM_YYYYMMDDHHmmss.zip
    Returns the target datetime (first timestamp)
    """
    import re
    match = re.search(r'PREDISPATCHIS_(\d{12})_', filename)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d%H%M')
        except:
            return None
    return None


def fetch_aemo_earlier_predispatch(target_time, log_callback=None):
    """Fetch earlier PreDispatch data that contains today's full forecast
    
    Args:
        target_time: datetime - we need a file that was published BEFORE this time
                     but contains forecast data covering this time
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    try:
        url = "https://nemweb.com.au/Reports/Current/PredispatchIS_Reports/"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        # Parse all files with their timestamps
        file_info = []
        for link in links:
            href = link['href']
            if href.endswith('.zip') and 'PREDISPATCHIS' in href.upper():
                filename = href.split('/')[-1]
                file_time = parse_predispatch_filename(filename)
                if file_time:
                    file_info.append({
                        'href': href,
                        'filename': filename,
                        'target_time': file_time
                    })
        
        if len(file_info) < 2:
            log("Not enough files available")
            return None
        
        # Sort by target time
        file_info.sort(key=lambda x: x['target_time'])
        
        log(f"Found {len(file_info)} files on server")
        log(f"  Earliest: {file_info[0]['target_time']}")
        log(f"  Latest: {file_info[-1]['target_time']}")
        
        # Find a file that:
        # 1. Has target_time at least 20 hours before our target (to have enough historical coverage)
        # 2. But not too old (within last 48 hours)
        
        ideal_file_time = target_time - timedelta(hours=20)
        min_acceptable_time = target_time - timedelta(hours=48)
        
        # Find the best matching file
        best_file = None
        for info in file_info:
            if min_acceptable_time <= info['target_time'] <= ideal_file_time:
                # Prefer the one closest to ideal_file_time
                if best_file is None or info['target_time'] > best_file['target_time']:
                    best_file = info
        
        # If no ideal file found, try to find any earlier file
        if best_file is None:
            for info in file_info:
                if info['target_time'] < target_time - timedelta(hours=12):
                    if best_file is None or info['target_time'] > best_file['target_time']:
                        best_file = info
        
        if best_file is None:
            log("Could not find suitable historical file")
            return None
        
        earlier_zip = best_file['href']
        if earlier_zip.startswith('/'):
            zip_url = 'https://nemweb.com.au' + earlier_zip
        else:
            zip_url = url + earlier_zip
        
        log(f"Selected: {best_file['filename']}")
        log(f"  Target time: {best_file['target_time']}")
        
        zip_response = requests.get(zip_url, timeout=60)
        zip_response.raise_for_status()
        
        zip_buffer = io.BytesIO(zip_response.content)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            csv_files = [name for name in zip_file.namelist() if name.endswith('.CSV')]
            if not csv_files:
                return None
            
            with zip_file.open(csv_files[0]) as csv_file:
                csv_content = csv_file.read().decode('utf-8')
        
        lines = csv_content.strip().split('\n')
        
        header_line = None
        header_idx = -1
        table_prefix = None
        
        for idx, line in enumerate(lines):
            if 'REGION_PRICES' in line and line.startswith('I,'):
                header_line = line
                header_idx = idx
                parts = line.split(',')
                if len(parts) >= 2:
                    table_prefix = parts[1]
                break
        
        if header_line is None:
            return None
        
        columns = [col.strip() for col in header_line.split(',')]
        
        try:
            regionid_idx = columns.index('REGIONID')
            datetime_idx = columns.index('DATETIME')
            rrp_idx = columns.index('RRP')
        except:
            return None
        
        data_rows = []
        data_line_prefix = f'D,{table_prefix},REGION_PRICES'
        
        for line in lines[header_idx + 1:]:
            if line.startswith(data_line_prefix):
                parts = [part.strip() for part in line.split(',')]
                
                if len(parts) > regionid_idx and parts[regionid_idx] == 'NSW1':
                    try:
                        datetime_str = parts[datetime_idx].strip('"')
                        price_str = parts[rrp_idx].strip('"')
                        
                        dt = pd.to_datetime(datetime_str, format='%Y/%m/%d %H:%M:%S')
                        price = float(price_str)
                        
                        data_rows.append({'DateTime': dt, 'Price': price})
                    except:
                        continue
        
        if not data_rows:
            return None
        
        df = pd.DataFrame(data_rows)
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        log(f"Fetched {len(df)} records from historical file")
        log(f"  Time range: {df['DateTime'].min()} ~ {df['DateTime'].max()}")
        return df
        
    except Exception as e:
        log(f"Failed to fetch historical data: {e}")
        return None


def fetch_combined_aemo_data(log_callback=None):
    """Fetch and combine AEMO data (today + tomorrow)"""
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    now = get_sydney_now()
    # Remove timezone info for comparison with AEMO data (which is timezone-naive but in AEST/AEDT)
    now_naive = now.replace(tzinfo=None)
    log(f"Current time (Sydney): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get latest forecast
    log("\n[1/2] Fetching latest forecast data...")
    df_latest = fetch_aemo_predispatch(log_callback)
    if df_latest is None:
        return None, None
    
    log(f"  Latest data range: {df_latest['DateTime'].min()} ~ {df_latest['DateTime'].max()}")
    
    # Determine if we need historical data
    today_start = datetime.combine(now_naive.date(), datetime.min.time())
    earliest_latest = df_latest['DateTime'].min()
    
    # Check if latest file already covers today from the start
    needs_historical = earliest_latest > today_start + timedelta(hours=1)
    
    df_earlier = None
    if needs_historical:
        log("\n[2/2] Fetching historical forecast data...")
        log(f"  Need data from: {today_start}")
        # Target: find a file that covers today's start
        df_earlier = fetch_aemo_earlier_predispatch(today_start, log_callback)
    else:
        log("\n[2/2] Latest file already covers today's start - skipping historical fetch")
    
    # Merge data
    log("\nMerging data...")
    if df_earlier is not None and len(df_earlier) > 0:
        df_combined = pd.concat([df_earlier, df_latest], ignore_index=True)
        # Keep latest data when duplicates (more accurate)
        df_combined = df_combined.drop_duplicates(subset=['DateTime'], keep='last')
        log(f"  Combined {len(df_earlier)} historical + {len(df_latest)} latest records")
    else:
        df_combined = df_latest.copy()
        if needs_historical:
            log("  Warning: Could not fetch historical data, using latest only")
    
    df_combined = df_combined.sort_values('DateTime').reset_index(drop=True)
    
    # Mark type (use timezone-naive now for comparison)
    df_combined['Type'] = df_combined['DateTime'].apply(
        lambda x: 'historical' if x < now_naive else 'forecast'
    )
    
    # Keep only today and tomorrow
    today = now_naive.date()
    tomorrow = today + timedelta(days=1)
    df_combined['Date'] = df_combined['DateTime'].dt.date
    df_two_days = df_combined[df_combined['Date'].isin([today, tomorrow])].copy()
    
    if len(df_two_days) == 0:
        log("Error: No data for today or tomorrow")
        return None, None
    
    # Analysis
    historical_count = len(df_two_days[df_two_days['Type'] == 'historical'])
    forecast_count = len(df_two_days[df_two_days['Type'] == 'forecast'])
    total_hours = len(df_two_days) * 0.5  # 30-min intervals
    
    log(f"\n✓ Final data statistics:")
    log(f"  Historical: {historical_count} records")
    log(f"  Forecast: {forecast_count} records")
    log(f"  Total: {len(df_two_days)} records ({total_hours:.1f} hours)")
    log(f"  Time range: {df_two_days['DateTime'].min()} ~ {df_two_days['DateTime'].max()}")
    
    # Warning if data seems incomplete
    if total_hours < 40:
        log(f"\n⚠ Warning: Data coverage ({total_hours:.1f}h) is less than expected 48h")
        log("  Optimization will still run with available data")
    
    return df_two_days[['DateTime', 'Price', 'Type']], now_naive


# ============== Optimization Function ==============

def run_optimization(df_price, params, daily_targets, log_callback=None):
    """Run pump scheduling optimization"""
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    log("Starting optimization...")
    
    df_price = df_price.copy()
    df_price["DateTime"] = pd.to_datetime(df_price["DateTime"])
    df_price = df_price.sort_values("DateTime").reset_index(drop=True)
    
    if len(df_price) < 2:
        raise ValueError("Insufficient data points")
    
    time_delta = df_price['DateTime'].iloc[1] - df_price['DateTime'].iloc[0]
    time_step_min = time_delta.total_seconds() / 60
    time_step_hr = time_step_min / 60.0
    log(f"Time interval: {time_step_min:.0f} minutes")
    
    # Parameters
    Q_on = params['q_on']
    P_on = params['p_on']
    P_standby = params['p_standby']
    min_on_hours = params['min_on_hours']
    daily_min_hours = params['daily_min_hours']
    default_daily_target = params['daily_target']
    
    T = len(df_price)
    Q_step_ML = Q_on * (time_step_min * 60) / 1e6
    
    df_price['Date'] = df_price['DateTime'].dt.date
    
    # Date mapping
    day_mapping = {}
    for t in range(T):
        date = df_price['Date'].iloc[t]
        if date not in day_mapping:
            day_mapping[date] = []
        day_mapping[date].append(t)
    
    log(f"Total {len(day_mapping)} days of data")
    
    # Restricted time slots (4PM-8PM on weekdays)
    restricted_slots = []
    for t in range(T):
        dt = df_price['DateTime'].iloc[t]
        if dt.weekday() < 5:
            hour = dt.hour
            if 16 <= hour < 20:
                restricted_slots.append(t)
    
    log(f"Restricted slots: {len(restricted_slots)}")
    
    # Build model
    log("Building optimization model...")
    model = pulp.LpProblem("Pump_Scheduler", pulp.LpMinimize)
    u = pulp.LpVariable.dicts("u", range(T), cat='Binary')
    v = pulp.LpVariable.dicts("v", range(T), cat='Binary')
    
    # Objective function
    model += pulp.lpSum([
        (df_price['Price'].iloc[t] * (P_on / 1000) * u[t] + 
         df_price['Price'].iloc[t] * (P_standby / 1000) * (1 - u[t])) * time_step_hr 
        for t in range(T)
    ]), "Total_Cost"
    
    # Constraints
    model += v[0] >= u[0]
    for t in range(1, T):
        model += v[t] >= u[t] - u[t-1]
    
    for t in restricted_slots:
        model += u[t] == 0
    
    # Minimum continuous run time
    min_on_steps = int(min_on_hours * 60 / time_step_min)
    
    for t in range(max(0, T - min_on_steps + 1), T):
        model += v[t] == 0
    
    for t in range(T - min_on_steps + 1):
        for i in range(min_on_steps):
            model += u[t + i] >= v[t]
    
    # Daily constraints
    daily_min_steps = int(daily_min_hours * 60 / time_step_min)
    
    for date, timesteps in day_mapping.items():
        # Get target for this specific day
        date_str = date.strftime('%Y-%m-%d')
        target_ml = daily_targets.get(date_str, default_daily_target)
        
        model += pulp.lpSum([u[t] for t in timesteps]) >= daily_min_steps
        model += pulp.lpSum([u[t] * Q_step_ML for t in timesteps]) >= target_ml
        log(f"  {date_str}: Target = {target_ml:.1f} ML")
    
    # Solve
    log("Solving...")
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[model.status] != 'Optimal':
        raise Exception(f"Optimization failed: {pulp.LpStatus[model.status]}")
    
    log("Solution found!")
    
    # Results
    u_sol = np.array([u[t].value() for t in range(T)])
    
    # Calculate statistics
    total_cost = sum(
        (df_price['Price'].iloc[t] * (P_on / 1000) * u_sol[t] + 
         df_price['Price'].iloc[t] * (P_standby / 1000) * (1 - u_sol[t])) * time_step_hr 
        for t in range(T)
    )
    
    total_volume = sum(u_sol) * Q_step_ML
    total_hours = sum(u_sol) * time_step_hr
    
    # Extract schedule with cost per period
    schedule_summary = []
    is_on = u_sol > 0.5
    start_index = -1
    
    def calc_period_cost(start_t, end_t):
        """Calculate cost for a period"""
        cost = 0
        for t in range(start_t, end_t + 1):
            cost += df_price['Price'].iloc[t] * (P_on / 1000) * time_step_hr
        return cost
    
    for t in range(T):
        if is_on[t] and (t == 0 or not is_on[t-1]):
            start_index = t
        if not is_on[t] and t > 0 and is_on[t-1]:
            end_index = t - 1
            duration = (end_index - start_index + 1) * time_step_hr
            volume = (end_index - start_index + 1) * Q_step_ML
            cost = calc_period_cost(start_index, end_index)
            schedule_summary.append({
                "Start Time": df_price['DateTime'].iloc[start_index],
                "Stop Time": df_price['DateTime'].iloc[t],
                "Duration (h)": duration,
                "Volume (ML)": volume,
                "Cost ($)": cost,
                "Unit Cost ($/ML)": cost / volume if volume > 0 else 0
            })
            start_index = -1
    
    if start_index != -1:
        end_index = T - 1
        duration = (end_index - start_index + 1) * time_step_hr
        volume = (end_index - start_index + 1) * Q_step_ML
        cost = calc_period_cost(start_index, end_index)
        schedule_summary.append({
            "Start Time": df_price['DateTime'].iloc[start_index],
            "Stop Time": df_price['DateTime'].iloc[end_index] + pd.Timedelta(minutes=time_step_min),
            "Duration (h)": duration,
            "Volume (ML)": volume,
            "Cost ($)": cost,
            "Unit Cost ($/ML)": cost / volume if volume > 0 else 0
        })
    
    # Calculate daily summary
    daily_summary = {}
    for date, timesteps in day_mapping.items():
        day_cost = sum(
            (df_price['Price'].iloc[t] * (P_on / 1000) * u_sol[t] + 
             df_price['Price'].iloc[t] * (P_standby / 1000) * (1 - u_sol[t])) * time_step_hr 
            for t in timesteps
        )
        day_volume = sum(u_sol[t] for t in timesteps) * Q_step_ML
        day_hours = sum(u_sol[t] for t in timesteps) * time_step_hr
        daily_summary[date] = {
            'cost': day_cost,
            'volume': day_volume,
            'hours': day_hours,
            'unit_cost': day_cost / day_volume if day_volume > 0 else 0
        }
    
    # Add pump status to df_price for export
    df_price['Pump_Status'] = ['ON' if u > 0.5 else 'OFF' for u in u_sol]
    
    return {
        'df_price': df_price,
        'u_sol': u_sol,
        'schedule': schedule_summary,
        'daily_summary': daily_summary,
        'total_cost': total_cost,
        'total_volume': total_volume,
        'total_hours': total_hours
    }


# ============== Chart Function ==============

def create_schedule_chart(df_price, u_sol, current_time=None):
    """Create interactive schedule chart"""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    has_type = 'Type' in df_price.columns
    
    if has_type:
        # Historical price
        df_hist = df_price[df_price['Type'] == 'historical']
        if len(df_hist) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_hist['DateTime'],
                    y=df_hist['Price'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#1f77b4', width=2, dash='dash'),
                    opacity=0.6,
                    hovertemplate='<b>Historical</b><br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
                ),
                secondary_y=False
            )
        
        # Forecast price
        df_fore = df_price[df_price['Type'] == 'forecast']
        if len(df_fore) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_fore['DateTime'],
                    y=df_fore['Price'],
                    mode='lines',
                    name='Forecast Price',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>Forecast</b><br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
                ),
                secondary_y=False
            )
        
        # Current time line
        if current_time:
            fig.add_shape(
                type="line",
                x0=current_time,
                x1=current_time,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color='red', width=2, dash='solid')
            )
            fig.add_annotation(
                x=current_time,
                y=1,
                yref="paper",
                text="NOW",
                showarrow=False,
                font=dict(color='red', size=12),
                yshift=10
            )
        
        # Historical pump status (step chart)
        if len(df_hist) > 0:
            hist_mask = df_price['Type'] == 'historical'
            hist_indices = df_price[hist_mask].index.tolist()
            
            fig.add_trace(
                go.Scatter(
                    x=df_price.loc[hist_indices, 'DateTime'],
                    y=[u_sol[i] for i in hist_indices],
                    mode='lines',
                    name='Historical Status',
                    fill='tozeroy',
                    line=dict(color='#888888', width=1, shape='hv'),
                    fillcolor='rgba(136, 136, 136, 0.3)',
                    hovertemplate='<b>Historical</b><br>Time: %{x}<br>Pump: %{customdata}<extra></extra>',
                    customdata=['ON' if u_sol[i] > 0.5 else 'OFF' for i in hist_indices]
                ),
                secondary_y=True
            )
        
        # Forecast pump status (step chart)
        if len(df_fore) > 0:
            fore_mask = df_price['Type'] == 'forecast'
            fore_indices = df_price[fore_mask].index.tolist()
            
            fig.add_trace(
                go.Scatter(
                    x=df_price.loc[fore_indices, 'DateTime'],
                    y=[u_sol[i] for i in fore_indices],
                    mode='lines',
                    name='Scheduled Status',
                    fill='tozeroy',
                    line=dict(color='#ff7f0e', width=1, shape='hv'),
                    fillcolor='rgba(255, 127, 14, 0.6)',
                    hovertemplate='<b>Scheduled</b><br>Time: %{x}<br>Pump: %{customdata}<extra></extra>',
                    customdata=['ON' if u_sol[i] > 0.5 else 'OFF' for i in fore_indices]
                ),
                secondary_y=True
            )
    else:
        # Single price curve
        fig.add_trace(
            go.Scatter(
                x=df_price['DateTime'],
                y=df_price['Price'],
                mode='lines',
                name='Price ($/MWh)',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Pump status (step chart)
        fig.add_trace(
            go.Scatter(
                x=df_price['DateTime'],
                y=u_sol,
                mode='lines',
                name='Pump Status',
                fill='tozeroy',
                line=dict(color='#ff7f0e', width=1, shape='hv'),
                fillcolor='rgba(255, 127, 14, 0.6)',
                hovertemplate='Time: %{x}<br>Pump: %{customdata}<extra></extra>',
                customdata=['ON' if s > 0.5 else 'OFF' for s in u_sol]
            ),
            secondary_y=True
        )
    
    # Layout
    fig.update_layout(
        title='Pump Schedule Optimization Result',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig.update_xaxes(title_text="Date and Time")
    fig.update_yaxes(title_text="Price ($/MWh)", secondary_y=False, color='#1f77b4')
    fig.update_yaxes(
        title_text="Pump Status",
        secondary_y=True,
        color='#ff7f0e',
        tickvals=[0, 1],
        ticktext=['OFF', 'ON'],
        range=[-0.1, 1.1]
    )
    
    return fig


# ============== Streamlit Main App ==============

def main():
    # Header row with title and logos side by side
    import base64
    
    # Title row
    st.markdown("""
    <div style="text-align: center; margin-bottom: 10px;">
        <h1 style="font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin: 0;">🔌 Pump Scheduling Optimizer</h1>
        <p style="font-size: 1rem; color: #666; margin: 5px 0 15px 0;">Economic Optimization Based on Price Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logos row - centered, larger size
    logo_images = ""
    for fig_name in ["fig1.png", "fig2.png", "fig3.png"]:
        try:
            with open(fig_name, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                logo_images += f'<img src="data:image/png;base64,{data}" style="width: 200px; height: auto; margin: 0 15px;" />'
        except:
            pass
    
    if logo_images:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            {logo_images}
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'df_price' not in st.session_state:
        st.session_state.df_price = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'current_time' not in st.session_state:
        st.session_state.current_time = None
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'available_dates' not in st.session_state:
        st.session_state.available_dates = []
    
    # Sidebar - Parameters
    with st.sidebar:
        st.header("⚙️ System Parameters")
        
        q_on = st.number_input(
            "Flow Rate (L/s)",
            min_value=0.0,
            max_value=5000.0,
            value=1050.0,
            step=10.0
        )
        
        p_on = st.number_input(
            "Max Power (kW)",
            min_value=0.0,
            max_value=10000.0,
            value=1950.0,
            step=10.0
        )
        
        p_standby = st.number_input(
            "Standby Power (kW)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )
        
        min_on_hours = st.number_input(
            "Min Continuous Run (h)",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )
        
        daily_min_hours = st.number_input(
            "Min Daily Run (h)",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )
        
        st.divider()
        st.header("📊 Data Source")
        
        data_source = st.radio(
            "Select data source:",
            ["AEMO Live Data", "Upload CSV File"],
            index=0
        )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📥 Data Input")
        
        if data_source == "AEMO Live Data":
            if st.button("🔄 Fetch AEMO Data", type="primary", use_container_width=True):
                with st.spinner("Fetching data from AEMO NEMWEB..."):
                    logs = []
                    def log_callback(msg):
                        logs.append(msg)
                    
                    df, current_time = fetch_combined_aemo_data(log_callback)
                    st.session_state.logs = logs
                    
                    if df is not None:
                        st.session_state.df_price = df
                        st.session_state.current_time = current_time
                        st.session_state.result = None
                        # Get available dates
                        df['Date'] = pd.to_datetime(df['DateTime']).dt.date
                        st.session_state.available_dates = sorted(df['Date'].unique())
                        st.success(f"Successfully fetched {len(df)} records")
                    else:
                        st.error("Failed to fetch data")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="CSV file must contain DateTime and Price columns"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) >= 2:
                        df.columns = ['DateTime', 'Price'] + list(df.columns[2:])
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    st.session_state.df_price = df
                    st.session_state.current_time = None
                    st.session_state.result = None
                    # Get available dates
                    df['Date'] = df['DateTime'].dt.date
                    st.session_state.available_dates = sorted(df['Date'].unique())
                    st.success(f"Successfully loaded {len(df)} records")
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
        
        # Data preview
        if st.session_state.df_price is not None:
            st.subheader(f"📋 Data Preview ({len(st.session_state.df_price)} records)")
            st.dataframe(
                st.session_state.df_price,
                use_container_width=True,
                height=300
            )
            
            with st.expander("📜 Fetch Logs"):
                for log in st.session_state.logs:
                    st.text(log)
        
        st.divider()
        
        # Daily targets section
        st.subheader("🎯 Daily Pumping Targets")
        
        default_target = st.number_input(
            "Default Target (ML/day)",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=1.0,
            key="default_target"
        )
        
        daily_targets = {}
        
        if st.session_state.available_dates:
            st.write("**Set target for each day:**")
            for date in st.session_state.available_dates:
                date_str = date.strftime('%Y-%m-%d')
                col_date, col_target = st.columns([1, 1])
                with col_date:
                    st.write(f"📅 {date_str}")
                with col_target:
                    target = st.number_input(
                        f"Target for {date_str}",
                        min_value=0.0,
                        max_value=200.0,
                        value=default_target,
                        step=1.0,
                        key=f"target_{date_str}",
                        label_visibility="collapsed"
                    )
                    daily_targets[date_str] = target
        
        st.divider()
        
        # Run optimization
        if st.session_state.df_price is not None:
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                params = {
                    'daily_target': default_target,
                    'q_on': q_on,
                    'p_on': p_on,
                    'p_standby': p_standby,
                    'min_on_hours': min_on_hours,
                    'daily_min_hours': daily_min_hours
                }
                
                logs = []
                def log_callback(msg):
                    logs.append(msg)
                
                with st.spinner("Optimizing..."):
                    try:
                        result = run_optimization(
                            st.session_state.df_price,
                            params,
                            daily_targets,
                            log_callback
                        )
                        st.session_state.result = result
                        st.session_state.logs = logs
                        st.success("Optimization complete!")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        st.session_state.logs = logs
    
    with col2:
        st.subheader("📈 Optimization Result & Schedule")
        
        if st.session_state.result is not None:
            result = st.session_state.result
            
            # Chart first
            fig = create_schedule_chart(
                result['df_price'],
                result['u_sol'],
                st.session_state.current_time
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily summary with schedules
            if result['daily_summary'] and result['schedule']:
                schedule_df = pd.DataFrame(result['schedule'])
                schedule_df['Date'] = pd.to_datetime(schedule_df['Start Time']).dt.date
                
                for date in sorted(result['daily_summary'].keys()):
                    daily = result['daily_summary'][date]
                    date_str = date.strftime('%Y-%m-%d (%A)')
                    
                    with st.expander(f"📅 {date_str}", expanded=True):
                        # Daily metrics
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Run Time", f"{daily['hours']:.1f} h")
                        with m2:
                            st.metric("Volume", f"{daily['volume']:.2f} ML")
                        with m3:
                            st.metric("Cost", f"${daily['cost']:.2f}")
                        with m4:
                            st.metric("Unit Cost", f"${daily['unit_cost']:.2f}/ML")
                        
                        # Schedule for this day
                        day_schedule = schedule_df[schedule_df['Date'] == date].copy()
                        if len(day_schedule) > 0:
                            # Format for display
                            display_df = day_schedule[['Start Time', 'Stop Time', 'Duration (h)', 'Volume (ML)', 'Cost ($)', 'Unit Cost ($/ML)']].copy()
                            display_df['Start Time'] = pd.to_datetime(display_df['Start Time']).dt.strftime('%H:%M')
                            display_df['Stop Time'] = pd.to_datetime(display_df['Stop Time']).dt.strftime('%H:%M')
                            display_df['Duration (h)'] = display_df['Duration (h)'].apply(lambda x: f"{x:.1f}")
                            display_df['Volume (ML)'] = display_df['Volume (ML)'].apply(lambda x: f"{x:.2f}")
                            display_df['Cost ($)'] = display_df['Cost ($)'].apply(lambda x: f"${x:.2f}")
                            display_df['Unit Cost ($/ML)'] = display_df['Unit Cost ($/ML)'].apply(lambda x: f"${x:.2f}")
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No pumping scheduled for this day")
            
            # Total summary
            st.divider()
            st.markdown("**📊 Total Summary**")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric("Total Run Time", f"{result['total_hours']:.1f} h")
            with t2:
                st.metric("Total Volume", f"{result['total_volume']:.2f} ML")
            with t3:
                st.metric("Total Cost", f"${result['total_cost']:.2f}")
            with t4:
                unit_cost = result['total_cost'] / result['total_volume'] if result['total_volume'] > 0 else 0
                st.metric("Avg Unit Cost", f"${unit_cost:.2f}/ML")
            
            # Download buttons
            st.subheader("📥 Download Results")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Excel download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    if result['schedule']:
                        pd.DataFrame(result['schedule']).to_excel(
                            writer, sheet_name='Schedule', index=False
                        )
                    result['df_price'].to_excel(
                        writer, sheet_name='Detailed_Results', index=False
                    )
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_buffer,
                    file_name=f"pump_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_d2:
                # CSV download
                csv_data = result['df_price'].to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"pump_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Please fetch data and run optimization first")
            
            # Placeholder
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        height: 400px; 
                        border-radius: 10px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        color: #666;">
                <div style="text-align: center;">
                    <h3>📊 Chart will appear here</h3>
                    <p>Fetch data and run optimization to see results</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
