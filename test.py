
try:
        from fno_matplotlib_visualizer import create_fno_charts
        
        print(f"\nCreating matplotlib charts...")
        chart_files = create_fno_charts(filename, output_dir='./fno_charts')
        
        if chart_files:
            print(f"Charts saved to: ./fno_charts/")
        else:
            print("Could not create charts")
            
    except ImportError:
        print("To create matplotlib charts, save the visualizer as 'fno_matplotlib_visualizer.py'")
    except Exception as e:
        print(f"Chart creation failed: {e}")