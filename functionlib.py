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


def fetch_aemo_earlier_predispatch(today_start, current_time, log_callback=None):
    """Fetch earlier PreDispatch data that contains today's full forecast

    Args:
        today_start: datetime - start of today (00:00)
        current_time: datetime - current time (we need to fill the gap from today_start to current_time)
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

        # Strategy: We need a file whose forecast covers from today_start to current_time
        # Each file contains ~40 hours of forecast from its target_time
        # So we need: file_target_time + 40h > current_time
        # AND: file_target_time <= today_start (to cover from today's beginning)

        # Ideal: find a file with target_time just before today_start (e.g., yesterday 23:30)
        # This file would forecast from yesterday 23:30 to day-after-tomorrow ~15:30

        log(f"  Looking for file to cover: {today_start} ~ {current_time}")

        # Find files that can cover current_time (target_time + 40h > current_time)
        # AND have target_time before or near today_start
        candidates = []
        for info in file_info:
            forecast_end = info['target_time'] + timedelta(hours=40)
            # Must cover current_time
            if forecast_end > current_time:
                # Prefer files with target_time close to but before today_start
                gap_to_today = (today_start - info['target_time']).total_seconds() / 3600
                if -2 <= gap_to_today <= 24:  # target_time within 24h before or 2h after today_start
                    candidates.append({
                        **info,
                        'gap_hours': gap_to_today,
                        'forecast_end': forecast_end
                    })

        if not candidates:
            # Fallback: any file that can reach current_time
            log("  No ideal candidate, trying fallback...")
            for info in file_info:
                forecast_end = info['target_time'] + timedelta(hours=40)
                if forecast_end > current_time and info['target_time'] < current_time:
                    candidates.append({
                        **info,
                        'gap_hours': (today_start - info['target_time']).total_seconds() / 3600,
                        'forecast_end': forecast_end
                    })

        if not candidates:
            log("Could not find suitable historical file")
            return None

        # Sort by gap_hours: prefer files with target_time just before today_start (gap ~0-2h)
        candidates.sort(key=lambda x: abs(x['gap_hours'] - 1))  # Prefer ~1 hour before today
        best_file = candidates[0]

        earlier_zip = best_file['href']
        if earlier_zip.startswith('/'):
            zip_url = 'https://nemweb.com.au' + earlier_zip
        else:
            zip_url = url + earlier_zip

        log(f"Selected: {best_file['filename']}")
        log(f"  Target time: {best_file['target_time']}")
        log(f"  Forecast covers until: ~{best_file['forecast_end']}")

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
        log(f"  Need to fill gap: {today_start} ~ {now_naive}")
        # Target: find a file that covers from today's start to now
        df_earlier = fetch_aemo_earlier_predispatch(today_start, now_naive, log_callback)
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

    # Check each day's data coverage and remove incomplete days
    daily_min_steps = int(daily_min_hours * 60 / time_step_min)
    min_on_steps = int(min_on_hours * 60 / time_step_min)

    # Calculate minimum required steps per day (considering constraints)
    # Need at least: daily_min_steps OR enough time to run min_on_hours
    min_required_steps = max(daily_min_steps, min_on_steps)
    min_required_hours = min_required_steps * time_step_hr

    days_to_remove = []
    for date, timesteps in day_mapping.items():
        available_hours = len(timesteps) * time_step_hr

        # Count restricted slots for this day (4PM-8PM on weekdays)
        restricted_count = 0
        for t in timesteps:
            dt = df_price['DateTime'].iloc[t]
            if dt.weekday() < 5 and 16 <= dt.hour < 20:
                restricted_count += 1

        usable_slots = len(timesteps) - restricted_count
        usable_hours = usable_slots * time_step_hr

        # Check if this day has enough usable time
        if usable_hours < daily_min_hours:
            log(f"⚠ Removing {date}: Only {available_hours:.1f}h data ({usable_hours:.1f}h usable), need {daily_min_hours:.1f}h minimum")
            days_to_remove.append(date)

    # Remove incomplete days from data
    if days_to_remove:
        log(f"Removing {len(days_to_remove)} incomplete day(s) from optimization")
        df_price = df_price[~df_price['Date'].isin(days_to_remove)].reset_index(drop=True)

        if len(df_price) == 0:
            raise Exception(
                "No complete days available for optimization. Please try again later when more forecast data is available.")

        # Rebuild T and day_mapping
        T = len(df_price)
        day_mapping = {}
        for t in range(T):
            date = df_price['Date'].iloc[t]
            if date not in day_mapping:
                day_mapping[date] = []
            day_mapping[date].append(t)

        log(f"Remaining: {len(day_mapping)} day(s), {T} time steps ({T * time_step_hr:.1f}h)")

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
        model += v[t] >= u[t] - u[t - 1]

    for t in restricted_slots:
        model += u[t] == 0

    # Minimum continuous run time constraint
    for t in range(max(0, T - min_on_steps + 1), T):
        model += v[t] == 0

    for t in range(T - min_on_steps + 1):
        for i in range(min_on_steps):
            model += u[t + i] >= v[t]

    # Daily constraints
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
        if is_on[t] and (t == 0 or not is_on[t - 1]):
            start_index = t
        if not is_on[t] and t > 0 and is_on[t - 1]:
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