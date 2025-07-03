"""
FNO Results Matplotlib Visualizer
A separate file to create professional matplotlib charts from your FNO training results.

Usage:
    After running your main script, use this to create visualizations:
    
    from fno_matplotlib_visualizer import create_fno_charts
    create_fno_charts('./visualizer/benchmark_results.json')
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob
import os

class FNOChartCreator:
    """Creates professional matplotlib charts from FNO training results"""
    
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        self.figsize = figsize
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['grid.alpha'] = 0.3

    def load_results(self, json_path):
        """Load results from JSON file"""
        with open(json_path, 'r') as f:
            return json.load(f)

    def create_training_curves(self, results_data, save_path=None):
        """Create training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get data
        training_history = results_data.get('training_history', {})
        accuracy_history = results_data.get('accuracy_history', {})
        
        # Plot 1: Loss curves
        if training_history:
            for i, (model_name, loss_values) in enumerate(training_history.items()):
                color = self.colors[i % len(self.colors)]
                epochs = range(1, len(loss_values) + 1)
                clean_name = model_name.replace('_64x64', '')
                
                ax1.plot(epochs, loss_values, color=color, linewidth=2, 
                        label=clean_name, alpha=0.8)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('Training Loss Curves')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # Plot 2: Accuracy curves
        if accuracy_history:
            for i, (model_name, acc_data) in enumerate(accuracy_history.items()):
                color = self.colors[i % len(self.colors)]
                clean_name = model_name.replace('_64x64', '')
                
                # Plot training accuracy
                train_acc = acc_data.get('train', [])
                if train_acc:
                    epochs = range(1, len(train_acc) + 1)
                    ax2.plot(epochs, train_acc, color=color, linewidth=2, 
                            label=f'{clean_name} (train)', alpha=0.8)
                
                # Plot validation accuracy
                val_acc = acc_data.get('validation', [])
                if val_acc:
                    epochs = range(1, len(val_acc) + 1)
                    ax2.plot(epochs, val_acc, color=color, linewidth=2, 
                            linestyle='--', label=f'{clean_name} (val)', alpha=0.8)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training & Validation Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim([75, 95])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training curves saved: {save_path}")
        
        return fig

    def create_individual_charts(self, results_data, output_dir):
        """Create individual chart files for HTML display"""
        models = results_data.get('models', [])
        if not models:
            return {}
        
        output_path = Path(output_dir)
        chart_files = {}
        
        # Extract data
        names = [m['name'].replace('_64x64', '') for m in models]
        accuracies = [m['metrics']['accuracy'] for m in models]
        parameters = [m['model_info']['parameters'] for m in models]
        training_times = [m['metrics'].get('wall_sec', 0) / 60 for m in models]
        rel_l2_errors = [m['metrics']['relative_l2'] for m in models]
        
        # 1. Training Time vs Accuracy
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        scatter = ax.scatter(training_times, accuracies, c=range(len(names)), 
                           cmap='viridis', s=200, alpha=0.8, edgecolors='black', linewidth=2)
        ax.set_xlabel('Training Time (minutes)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add model labels
        for i, (time, acc, name) in enumerate(zip(training_times, accuracies, names)):
            ax.annotate(name, (time, acc), xytext=(8, 8), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        time_vs_acc_path = output_path / 'training_time_vs_accuracy.png'
        plt.savefig(time_vs_acc_path, dpi=300, bbox_inches='tight')
        chart_files['training_time_vs_accuracy'] = time_vs_acc_path
        plt.close()
        
        # 2. Relative L2 Error Chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(range(len(names)), rel_l2_errors, color=self.colors[:len(names)], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title('Relative L2 Error Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add error labels on bars
        for bar, err in zip(bars, rel_l2_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{err:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        rel_l2_path = output_path / 'relative_l2_error.png'
        plt.savefig(rel_l2_path, dpi=300, bbox_inches='tight')
        chart_files['relative_l2_error'] = rel_l2_path
        plt.close()
        
        # 3. Model Efficiency Chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        efficiency = [acc / (param / 1000) for acc, param in zip(accuracies, parameters)]
        bars = ax.bar(range(len(names)), efficiency, color=self.colors[:len(names)], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy per 1000 Parameters', fontsize=12)
        ax.set_title('Model Efficiency (Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add efficiency labels on bars
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{eff:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        efficiency_path = output_path / 'model_efficiency.png'
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        chart_files['model_efficiency'] = efficiency_path
        plt.close()
        
        return chart_files

    def create_model_comparison(self, results_data, save_path=None):
        """Create comprehensive model comparison chart"""
        models = results_data.get('models', [])
        if not models:
            return None
            
        # Extract data
        names = [m['name'].replace('_64x64', '') for m in models]
        accuracies = [m['metrics']['accuracy'] for m in models]
        parameters = [m['model_info']['parameters'] for m in models]
        training_times = [m['metrics'].get('wall_sec', 0) / 60 for m in models]
        rel_l2_errors = [m['metrics']['relative_l2'] for m in models]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Final Accuracy
        bars = ax1.bar(range(len(names)), accuracies, color=self.colors[:len(names)], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Final Model Accuracy')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameters vs Accuracy
        ax2.scatter(parameters, accuracies, c=range(len(names)), 
                   cmap='viridis', s=150, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Parameters vs Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Time vs Accuracy
        ax3.scatter(training_times, accuracies, c=range(len(names)), 
                   cmap='viridis', s=150, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Training Time (minutes)')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Training Time vs Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Relative L2 Error
        bars2 = ax4.bar(range(len(names)), rel_l2_errors, color=self.colors[:len(names)], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Relative L2 Error')
        ax4.set_title('Relative L2 Error (Lower is Better)')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison saved: {save_path}")
        
        return fig

    def create_architecture_analysis(self, results_data, save_path=None):
        """Create architecture parameter analysis"""
        models = results_data.get('models', [])
        
        # Extract architecture data
        arch_data = []
        for model in models:
            if 'architecture' in model['model_info']:
                arch = model['model_info']['architecture']
                arch_data.append({
                    'name': model['name'].replace('_64x64', ''),
                    'modes': arch.get('modes', 0),
                    'width': arch.get('width', 0),
                    'layers': arch.get('layers', 0),
                    'accuracy': model['metrics']['accuracy'],
                    'parameters': model['model_info']['parameters'],
                    'share_weights': arch.get('share_weights', False)
                })
        
        if not arch_data:
            return None
        
        df = pd.DataFrame(arch_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Modes vs Accuracy
        scatter1 = ax1.scatter(df['modes'], df['accuracy'], c=df['parameters'], 
                             cmap='viridis', s=150, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Fourier Modes')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Fourier Modes vs Accuracy')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Parameters')
        
        # Plot 2: Width vs Accuracy
        colors = ['red' if sw else 'blue' for sw in df['share_weights']]
        ax2.scatter(df['width'], df['accuracy'], c=colors, s=150, 
                   alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Network Width')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Width vs Accuracy\n(Red=Shared Weights, Blue=Independent)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Efficiency (Accuracy per 1000 parameters)
        efficiency = df['accuracy'] / (df['parameters'] / 1000)
        bars = ax3.bar(range(len(df)), efficiency, color=self.colors[:len(df)])
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Accuracy per 1000 Parameters')
        ax3.set_title('Model Efficiency (Higher is Better)')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['name'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Architecture Space Overview
        ax4.scatter(df['modes'], df['width'], c=df['accuracy'], s=df['parameters']/100, 
                   cmap='RdYlGn', alpha=0.7, edgecolors='black')
        ax4.set_xlabel('Fourier Modes')
        ax4.set_ylabel('Network Width')
        ax4.set_title('Architecture Space\n(Color=Accuracy, Size=Parameters)')
        ax4.grid(True, alpha=0.3)
        
        # Add model labels for all plots
        for _, row in df.iterrows():
            ax1.annotate(row['name'], (row['modes'], row['accuracy']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
            ax2.annotate(row['name'], (row['width'], row['accuracy']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
            ax4.annotate(row['name'], (row['modes'], row['width']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Architecture analysis saved: {save_path}")
        
        return fig

    def create_summary_table(self, results_data, save_path=None):
        """Create and save summary table"""
        models = results_data.get('models', [])
        
        data = []
        for model in models:
            data.append({
                'Model': model['name'].replace('_64x64', ''),
                'Accuracy (%)': f"{model['metrics']['accuracy']:.2f}",
                'Relative L2': f"{model['metrics']['relative_l2']:.4f}",
                'Parameters': f"{model['model_info']['parameters']:,}",
                'Training Time (min)': f"{model['metrics'].get('wall_sec', 0)/60:.1f}",
                'MAE': f"{model['metrics']['mae']:.6f}",
                'MSE': f"{model['metrics']['mse']:.6f}"
            })
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"📋 Summary table saved: {save_path}")
        
        return df


def create_fno_charts(json_file_path, output_dir='./fno_charts', show_plots=False):
    """
    Main function to create all FNO charts from your JSON results file
    
    Args:
        json_file_path: Path to your JSON results file (e.g., 'results_fno_20250703_122132.json')
        output_dir: Directory to save charts (default: './fno_charts')
        show_plots: Whether to display plots (default: False for batch processing)
    
    Returns:
        dict: Paths to all created files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize chart creator
    chart_creator = FNOChartCreator()
    
    # Load results
    try:
        results = chart_creator.load_results(json_file_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file {json_file_path}")
        return None
    
    print(f"🎨 Creating FNO charts from: {json_file_path}")
    print(f"📁 Saving to: {output_path}")
    
    created_files = {}
    
    # 1. Training curves (with fixed filename for HTML)
    curves_path = output_path / "training_curves.png"
    fig1 = chart_creator.create_training_curves(results, save_path=curves_path)
    if fig1:
        created_files['training_curves'] = curves_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig1)
    
    # 2. Create individual charts for HTML display
    individual_charts = chart_creator.create_individual_charts(results, output_path)
    created_files.update(individual_charts)
    
    # 3. Create full comparison chart (optional)
    comparison_path = output_path / "full_model_comparison.png"
    fig2 = chart_creator.create_model_comparison(results, save_path=comparison_path)
    if fig2:
        created_files['full_model_comparison'] = comparison_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig2)
    
    # 4. Architecture analysis (optional)
    arch_path = output_path / "architecture_analysis.png"
    fig3 = chart_creator.create_architecture_analysis(results, save_path=arch_path)
    if fig3:
        created_files['architecture_analysis'] = arch_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig3)
    
    # 5. Summary table
    table_path = output_path / "summary_table.csv"
    summary_df = chart_creator.create_summary_table(results, save_path=table_path)
    if summary_df is not None:
        created_files['summary_table'] = table_path
    
    # Print results summary
    models = results.get('models', [])
    if models:
        best_model = max(models, key=lambda x: x['metrics']['accuracy'])
        print(f"\n🏆 Best Model: {best_model['name'].replace('_64x64', '')}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.2f}%")
        print(f"   Parameters: {best_model['model_info']['parameters']:,}")
        print(f"   Training Time: {best_model['metrics'].get('wall_sec', 0)/60:.1f} min")
    
    print(f"\n✅ Charts created successfully!")
    print(f"📊 Files for HTML:")
    html_charts = ['training_curves', 'training_time_vs_accuracy', 'relative_l2_error', 'model_efficiency']
    for chart_name in html_charts:
        if chart_name in created_files:
            print(f"   • {chart_name}: {created_files[chart_name].name}")
    
    return created_files


def create_charts_from_latest_results(results_dir='.', output_dir='./fno_charts'):
    """
    Convenience function to create charts from the most recent results file
    
    Args:
        results_dir: Directory containing results files (default: current directory)
        output_dir: Directory to save charts
    
    Returns:
        dict: Paths to created files, or None if no results found
    """
    # Find the most recent results file
    pattern = os.path.join(results_dir, 'results_fno_*.json')
    results_files = glob.glob(pattern)
    
    # Also look for benchmark_results.json
    benchmark_file = os.path.join(results_dir, 'benchmark_results.json')
    if os.path.exists(benchmark_file):
        results_files.append(benchmark_file)
    
    if not results_files:
        print(f"❌ No results files found in {results_dir}")
        print("   Looking for files matching: results_fno_*.json or benchmark_results.json")
        return None
    
    # Get the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    print(f"📊 Using most recent results: {os.path.basename(latest_file)}")
    
    return create_fno_charts(latest_file, output_dir)


def create_comprehensive_report(json_file_path, output_dir='./fno_results'):
    """
    Create a comprehensive report from your JSON results file
    
    Args:
        json_file_path: Path to your JSON results file
        output_dir: Directory to save all outputs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize chart creator
    chart_creator = FNOChartCreator()
    
    # Load results
    try:
        results_data = chart_creator.load_results(json_file_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file {json_file_path}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"📊 Creating comprehensive FNO training report...")
    print(f"📁 Output directory: {output_path}")
    
    created_files = {}
    
    # 1. Training curves
    curves_path = output_path / f"training_curves_{timestamp}.png"
    fig1 = chart_creator.create_training_curves(results_data, save_path=curves_path)
    if fig1:
        created_files['training_curves'] = curves_path
        plt.close(fig1)
    
    # 2. Final comparison
    comparison_path = output_path / f"model_comparison_{timestamp}.png"
    fig2 = chart_creator.create_model_comparison(results_data, save_path=comparison_path)
    if fig2:
        created_files['model_comparison'] = comparison_path
        plt.close(fig2)
    
    # 3. Architecture analysis
    arch_path = output_path / f"architecture_analysis_{timestamp}.png"
    fig3 = chart_creator.create_architecture_analysis(results_data, save_path=arch_path)
    if fig3:
        created_files['architecture_analysis'] = arch_path
        plt.close(fig3)
    
    # 4. Individual charts
    individual_charts = chart_creator.create_individual_charts(results_data, output_path)
    created_files.update(individual_charts)
    
    # 5. Summary table
    table_path = output_path / f"results_summary_{timestamp}.csv"
    summary_df = chart_creator.create_summary_table(results_data, save_path=table_path)
    if summary_df is not None:
        created_files['summary_table'] = table_path
    
    # Print summary
    print(f"\n🎯 Training Summary:")
    print(f"├─ Dataset: {results_data.get('dataset_info', {}).get('name', 'Unknown')}")
    print(f"├─ Training samples: {results_data.get('dataset_info', {}).get('n_train_samples', 'Unknown')}")
    print(f"├─ Test samples: {results_data.get('dataset_info', {}).get('n_test_samples', 'Unknown')}")
    print(f"├─ Epochs trained: {results_data.get('epochs', 'Unknown')}")
    print(f"└─ Models compared: {len(results_data.get('models', []))}")
    
    # Find best model
    models = results_data.get('models', [])
    if models:
        best_model = max(models, key=lambda x: x['metrics']['accuracy'])
        print(f"\n🏆 Best Model: {best_model['name'].replace('_64x64', '')}")
        print(f"├─ Accuracy: {best_model['metrics']['accuracy']:.2f}%")
        print(f"├─ Parameters: {best_model['model_info']['parameters']:,}")
        print(f"├─ Training time: {best_model['metrics'].get('wall_sec', 0)/60:.1f} minutes")
        print(f"└─ Relative L2 error: {best_model['metrics']['relative_l2']:.4f}")
    
    print(f"\n✅ Comprehensive report created successfully!")
    print(f"📈 All plots saved in: {output_path}")
    
    return output_path, summary_df


# Example usage function
def analyze_fno_results(json_file_path, output_dir='./fno_analysis'):
    """
    Convenience function to analyze your FNO results
    
    Args:
        json_file_path: Path to your JSON results file
        output_dir: Where to save the analysis
    
    Example:
        analyze_fno_results('./visualizer/benchmark_results.json')
    """
    chart_creator = FNOChartCreator(figsize=(14, 8))
    output_path, summary_df = create_comprehensive_report(json_file_path, output_dir)
    
    if summary_df is not None:
        print(f"\n📊 Quick Analysis:")
        print("="*50)
        print(summary_df[['Model', 'Accuracy (%)', 'Parameters', 'Training Time (min)']].to_string(index=False))
    
    return chart_creator, output_path, summary_df


if __name__ == "__main__":
    # Demo with your actual data structure
    print("🎨 FNO Results Matplotlib Visualizer")
    print("💡 Usage examples:")
    print("  create_fno_charts('./visualizer/benchmark_results.json')")
    print("  create_charts_from_latest_results('./visualizer')")
    print("  analyze_fno_results('./visualizer/benchmark_results.json')")
    print("📊 This will create comprehensive visualizations from your JSON results!")
    
    create_fno_charts('/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/visualizer/benchmark_results.json')