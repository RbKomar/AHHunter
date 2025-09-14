import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from database import DatabaseManager

def plot_price_history(item_name):
    """
    Connects to the database, retrieves the price history for a specific item,
    and plots it using matplotlib.
    """
    db_manager = DatabaseManager()
    history = db_manager.get_price_history(item_name)
    db_manager.close()

    if not history:
        print(f"No price history found for item: '{item_name}'")
        return

    df = pd.DataFrame(history, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(df['timestamp'], df['price'], marker='o', linestyle='-', markersize=4)
    
    ax.set_title(f"Price History for: {item_name.title()}", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (Coins)", fontsize=12)
    fig.autofmt_xdate() 
    plt.tight_layout()
    
    sanitized_item_name = "".join(x for x in item_name.replace(" ", "_") if x.isalnum() or x == "_")
    output_filename = f"price_history_{sanitized_item_name}.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        item_to_plot = ' '.join(sys.argv[1:])
        plot_price_history(item_to_plot.lower())
    else:
        print("Usage: python analysis.py <item_name>")
        plot_price_history("pink tarantula skin")
