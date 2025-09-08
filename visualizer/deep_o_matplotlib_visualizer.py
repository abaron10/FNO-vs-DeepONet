

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob
import os

class DeepONetChartCreator:
    
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        self.figsize = figsize
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#ff9896', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d3'
        ]
        
                              
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
                                                          
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2

    def load_results(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def create_training_curves(self, results_data, save_path=None, create_individual=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
                  
        training_history = results_data.get('training_history', {})
        accuracy_history = results_data.get('accuracy_history', {})
        
                             
        if training_history:
            for i, (model_name, loss_values) in enumerate(training_history.items()):
                color = self.colors[i % len(self.colors)]
                epochs = range(1, len(loss_values) + 1)
                clean_name = model_name.replace('_64x64', '').replace('DeepONet', 'DON')
                
                ax1.plot(epochs, loss_values, color=color, linewidth=2.5, 
                        label=clean_name, alpha=0.8)
            
            ax1.set_xlabel('Epoch', fontweight='bold')
            ax1.set_ylabel('Training Loss', fontweight='bold')
            ax1.set_title('DeepONet Training Loss Curves', fontsize=16, fontweight='bold')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)

                                            
        if accuracy_history:
            for i, (model_name, acc_data) in enumerate(accuracy_history.items()):
                color = self.colors[i % len(self.colors)]
                clean_name = model_name.replace('_64x64', '').replace('DeepONet', 'DON')
                
                                                      
                val_acc = acc_data.get('validation', [])
                if val_acc:
                    epochs = range(1, len(val_acc) + 1)
                    ax2.plot(epochs, val_acc, color=color, linewidth=2.5, 
                            label=clean_name, alpha=0.8)
            
            ax2.set_xlabel('Epoch', fontweight='bold')
            ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold')
            ax2.set_title('DeepONet Validation Accuracy Overview', fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)
            
                                          
            if accuracy_history:
                all_vals = []
                for acc_data in accuracy_history.values():
                    all_vals.extend(acc_data.get('validation', []))
                if all_vals:
                    min_acc = max(60, min(all_vals) - 5)
                    max_acc = min(100, max(all_vals) + 2)
                    ax2.set_ylim([min_acc, max_acc])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 DeepONet training curves saved: {save_path}")
        
                                                       
        if create_individual and accuracy_history:
            output_dir = Path(save_path).parent if save_path else Path('.')
            self.create_individual_accuracy_plots(accuracy_history, output_dir)
        
        return fig

    def create_individual_accuracy_plots(self, accuracy_history, output_dir):
        output_path = Path(output_dir)
        accuracy_dir = output_path / 'individual_accuracy_plots'
        accuracy_dir.mkdir(exist_ok=True)
        
        print(f"\n📈 Creating individual DeepONet accuracy plots in: {accuracy_dir}")
        
        for i, (model_name, acc_data) in enumerate(accuracy_history.items()):
            clean_name = model_name.replace('_64x64', '').replace('DeepONet', 'DON')
            
                                          
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
                      
            train_acc = acc_data.get('train', [])
            val_acc = acc_data.get('validation', [])
            
                                    
            if train_acc:
                epochs = range(1, len(train_acc) + 1)
                ax.plot(epochs, train_acc, color='#2E86AB', linewidth=3, 
                       label='Training', alpha=0.8, marker='o', markersize=2, 
                       markevery=max(1, len(epochs)//25))
            
                                      
            if val_acc:
                epochs = range(1, len(val_acc) + 1)
                ax.plot(epochs, val_acc, color='#A23B72', linewidth=3, 
                       label='Validation', alpha=0.8, linestyle='--',
                       marker='s', markersize=2, markevery=max(1, len(epochs)//25))
            
                     
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'DeepONet Training Progress: {clean_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='lower right')
            
                                             
            all_values = train_acc + val_acc
            if all_values:
                min_val = max(60, min(all_values) - 3)
                max_val = min(100, max(all_values) + 2)
                ax.set_ylim([min_val, max_val])
            
                                           
            if val_acc:
                final_val_acc = val_acc[-1]
                ax.annotate(f'Final: {final_val_acc:.2f}%', 
                           xy=(len(val_acc), final_val_acc),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen', alpha=0.8),
                           fontsize=11, fontweight='bold')
            
                                                
            if len(val_acc) > 30:
                window = min(20, len(val_acc)//5)
                trend = pd.Series(val_acc).rolling(window=window, center=True).mean()
                ax.plot(epochs, trend, color='orange', linewidth=2, 
                       alpha=0.6, label='Trend', linestyle=':')
                ax.legend(fontsize=11, loc='lower right')
            
            plt.tight_layout()
            
                                  
            filename = f"accuracy_{clean_name.replace(' ', '_').lower()}.png"
            save_path = accuracy_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ {clean_name}: {filename}")
        
                                     
        self.create_accuracy_grid_view(accuracy_history, accuracy_dir)

    def create_accuracy_grid_view(self, accuracy_history, output_dir):
        n_models = len(accuracy_history)
        if n_models == 0:
            return
        
                                   
        cols = min(3, n_models)                                
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, acc_data) in enumerate(accuracy_history.items()):
            ax = axes[idx]
            clean_name = model_name.replace('_64x64', '').replace('DeepONet', 'DON')
            
                       
            train_acc = acc_data.get('train', [])
            val_acc = acc_data.get('validation', [])
            
            if train_acc:
                epochs = range(1, len(train_acc) + 1)
                ax.plot(epochs, train_acc, color='#2E86AB', linewidth=2, 
                       label='Train', alpha=0.8)
            
            if val_acc:
                epochs = range(1, len(val_acc) + 1)
                ax.plot(epochs, val_acc, color='#A23B72', linewidth=2, 
                       label='Val', alpha=0.8, linestyle='--')
            
                     
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(clean_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
                                   
            all_vals = train_acc + val_acc
            if all_vals:
                min_val = max(60, min(all_vals) - 2)
                max_val = min(100, max(all_vals) + 1)
                ax.set_ylim([min_val, max_val])
        
                               
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('DeepONet Accuracy Comparison Grid', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'accuracy_grid_view.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Grid view: accuracy_grid_view.png")

    def create_individual_charts(self, results_data, output_dir):
        models = results_data.get('models', [])
        if not models:
            return {}
        
        output_path = Path(output_dir)
        chart_files = {}
        
                      
        names = [m['name'].replace('_64x64', '').replace('DeepONet', 'DON') for m in models]
        accuracies = [m['metrics']['accuracy'] for m in models]
        parameters = [m['model_info']['parameters'] for m in models]
        training_times = [m['metrics'].get('wall_sec', 0) / 60 for m in models]
        rel_l2_errors = [m['metrics']['relative_l2'] for m in models]
        
                                      
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        scatter = ax.scatter(training_times, accuracies, c=range(len(names)), 
                           cmap='viridis', s=200, alpha=0.8, edgecolors='black', linewidth=2)
        ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('DeepONet: Training Time vs Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
                          
        for i, (time, acc, name) in enumerate(zip(training_times, accuracies, names)):
            ax.annotate(name, (time, acc), xytext=(8, 8), 
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        plt.tight_layout()
        time_vs_acc_path = output_path / 'deeponet_training_time_vs_accuracy.png'
        plt.savefig(time_vs_acc_path, dpi=300, bbox_inches='tight')
        chart_files['training_time_vs_accuracy'] = time_vs_acc_path
        plt.close()
        
                                    
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(range(len(names)), rel_l2_errors, color=self.colors[:len(names)], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xlabel('DeepONet Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative L2 Error', fontsize=12, fontweight='bold')
        ax.set_title('DeepONet Relative L2 Error Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
                                  
        for bar, err in zip(bars, rel_l2_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{err:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        rel_l2_path = output_path / 'deeponet_relative_l2_error.png'
        plt.savefig(rel_l2_path, dpi=300, bbox_inches='tight')
        chart_files['relative_l2_error'] = rel_l2_path
        plt.close()
        
                                   
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        efficiency = [acc / (param / 1000) for acc, param in zip(accuracies, parameters)]
        bars = ax.bar(range(len(names)), efficiency, color=self.colors[:len(names)], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xlabel('DeepONet Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy per 1000 Parameters', fontsize=12, fontweight='bold')
        ax.set_title('DeepONet Model Efficiency (Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
                                       
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{eff:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        efficiency_path = output_path / 'deeponet_model_efficiency.png'
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        chart_files['model_efficiency'] = efficiency_path
        plt.close()
        
        return chart_files

    def create_model_comparison(self, results_data, save_path=None):
        models = results_data.get('models', [])
        if not models:
            return None
            
                      
        names = [m['name'].replace('_64x64', '').replace('DeepONet', 'DON') for m in models]
        accuracies = [m['metrics']['accuracy'] for m in models]
        parameters = [m['model_info']['parameters'] for m in models]
        training_times = [m['metrics'].get('wall_sec', 0) / 60 for m in models]
        rel_l2_errors = [m['metrics']['relative_l2'] for m in models]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
                                
        bars = ax1.bar(range(len(names)), accuracies, color=self.colors[:len(names)], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_xlabel('DeepONet Models', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Final DeepONet Model Accuracy', fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
                             
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
                                        
        scatter2 = ax2.scatter(parameters, accuracies, c=range(len(names)), 
                              cmap='viridis', s=150, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Number of Parameters', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('DeepONet Parameters vs Accuracy', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
                          
        for i, (param, acc, name) in enumerate(zip(parameters, accuracies, names)):
            ax2.annotate(name, (param, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
                                           
        scatter3 = ax3.scatter(training_times, accuracies, c=range(len(names)), 
                              cmap='viridis', s=150, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Training Time (minutes)', fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontweight='bold')
        ax3.set_title('DeepONet Training Time vs Accuracy', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
                          
        for i, (time, acc, name) in enumerate(zip(training_times, accuracies, names)):
            ax3.annotate(name, (time, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
                                   
        bars2 = ax4.bar(range(len(names)), rel_l2_errors, color=self.colors[:len(names)], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_xlabel('DeepONet Models', fontweight='bold')
        ax4.set_ylabel('Relative L2 Error', fontweight='bold')
        ax4.set_title('DeepONet Relative L2 Error (Lower is Better)', fontweight='bold')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
                          
        for bar, err in zip(bars2, rel_l2_errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{err:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 DeepONet model comparison saved: {save_path}")
        
        return fig

    def create_architecture_analysis(self, results_data, save_path=None):
        models = results_data.get('models', [])
        
                                            
        arch_data = []
        for model in models:
            if 'architecture' in model['model_info']:
                arch = model['model_info']['architecture']
                arch_data.append({
                    'name': model['name'].replace('_64x64', '').replace('DeepONet', 'DON'),
                    'n_sensors': arch.get('n_sensors', 0),
                    'hidden_size': arch.get('hidden_size', 0),
                    'num_layers': arch.get('num_layers', 0),
                    'activation': arch.get('activation', 'unknown'),
                    'sensor_strategy': arch.get('sensor_strategy', 'unknown'),
                    'accuracy': model['metrics']['accuracy'],
                    'parameters': model['model_info']['parameters'],
                    'relative_l2': model['metrics']['relative_l2']
                })
        
        if not arch_data:
            return None
        
        df = pd.DataFrame(arch_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
                                               
        scatter1 = ax1.scatter(df['n_sensors'], df['accuracy'], c=df['parameters'], 
                             cmap='viridis', s=150, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Number of Sensors', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('DeepONet: Sensors vs Accuracy', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Parameters', fontweight='bold')
        
                                         
                                  
        strategies = df['sensor_strategy'].unique()
        strategy_colors = {strategy: self.colors[i] for i, strategy in enumerate(strategies)}
        colors = [strategy_colors[strategy] for strategy in df['sensor_strategy']]
        
        ax2.scatter(df['hidden_size'], df['accuracy'], c=colors, s=150, 
                   alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Hidden Size', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('DeepONet: Hidden Size vs Accuracy', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
                                          
        for strategy, color in strategy_colors.items():
            ax2.scatter([], [], c=[color], s=100, label=strategy, alpha=0.7, edgecolors='black')
        ax2.legend(title='Sensor Strategy', fontsize=8)
        
                                                  
        efficiency = df['accuracy'] / (df['parameters'] / 1000)
        bars = ax3.bar(range(len(df)), efficiency, color=self.colors[:len(df)], alpha=0.8)
        ax3.set_xlabel('DeepONet Models', fontweight='bold')
        ax3.set_ylabel('Accuracy per 1000 Parameters', fontweight='bold')
        ax3.set_title('DeepONet Architecture Efficiency', fontweight='bold')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['name'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
                                                                      
        scatter4 = ax4.scatter(df['n_sensors'], df['hidden_size'], c=df['accuracy'], 
                              s=df['parameters']/100, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        ax4.set_xlabel('Number of Sensors', fontweight='bold')
        ax4.set_ylabel('Hidden Size', fontweight='bold')
        ax4.set_title('DeepONet Architecture Space\n(Color=Accuracy, Size=Parameters)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        cbar4 = plt.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Accuracy (%)', fontweight='bold')
        
                                        
        for _, row in df.iterrows():
            ax1.annotate(row['name'], (row['n_sensors'], row['accuracy']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
            ax2.annotate(row['name'], (row['hidden_size'], row['accuracy']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
            ax4.annotate(row['name'], (row['n_sensors'], row['hidden_size']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 DeepONet architecture analysis saved: {save_path}")
        
        return fig

    def create_architecture_space_chart(self, results_data, save_path=None):
        models = results_data.get('models', [])
        
                                            
        arch_data = []
        for model in models:
            if 'architecture' in model['model_info']:
                arch = model['model_info']['architecture']
                arch_data.append({
                    'name': model['name'].replace('_64x64', '').replace('DeepONet', 'DON'),
                    'n_sensors': arch.get('n_sensors', 0),
                    'hidden_size': arch.get('hidden_size', 0),
                    'accuracy': model['metrics']['accuracy'],
                    'parameters': model['model_info']['parameters'],
                    'sensor_strategy': arch.get('sensor_strategy', 'unknown')
                })
        
        if not arch_data:
            return None
        
        df = pd.DataFrame(arch_data)
        
                                                         
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
                                                    
        scatter = ax.scatter(df['n_sensors'], df['hidden_size'], c=df['accuracy'], 
                           s=df['parameters']/50, cmap='RdYlGn', alpha=0.7, 
                           edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Number of Sensors', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hidden Size', fontsize=14, fontweight='bold')
        ax.set_title('DeepONet Architecture Space\n(Color=Accuracy, Size=Parameters)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
                           
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Accuracy (%)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
                                                  
        for _, row in df.iterrows():
            ax.annotate(row['name'], (row['n_sensors'], row['hidden_size']), 
                       xytext=(8, 8), textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                       fontweight='bold')
        
                                          
        strategies = df['sensor_strategy'].unique()
        if len(strategies) > 1:
                                                        
            strategy_text = f"Sensor Strategies: {', '.join(strategies)}"
            ax.text(0.02, 0.98, strategy_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7))
        
                                
        x_margin = (df['n_sensors'].max() - df['n_sensors'].min()) * 0.1
        y_margin = (df['hidden_size'].max() - df['hidden_size'].min()) * 0.1
        ax.set_xlim(df['n_sensors'].min() - x_margin, df['n_sensors'].max() + x_margin)
        ax.set_ylim(df['hidden_size'].min() - y_margin, df['hidden_size'].max() + y_margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 DeepONet architecture space chart saved: {save_path}")
        
        return fig

    def create_sensor_strategy_analysis(self, results_data, save_path=None):
        models = results_data.get('models', [])
        
                                  
        strategy_data = {}
        for model in models:
            if 'architecture' in model['model_info']:
                strategy = model['model_info']['architecture'].get('sensor_strategy', 'unknown')
                if strategy not in strategy_data:
                    strategy_data[strategy] = []
                
                strategy_data[strategy].append({
                    'name': model['name'].replace('_64x64', '').replace('DeepONet', 'DON'),
                    'accuracy': model['metrics']['accuracy'],
                    'n_sensors': model['model_info']['architecture'].get('n_sensors', 0),
                    'parameters': model['model_info']['parameters']
                })
        
        if not strategy_data:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
                                             
        strategies = list(strategy_data.keys())
        strategy_accuracies = []
        strategy_colors = []
        
        for i, strategy in enumerate(strategies):
            accuracies = [model['accuracy'] for model in strategy_data[strategy]]
            strategy_accuracies.append(accuracies)
            strategy_colors.append(self.colors[i % len(self.colors)])
        
                  
        bp = ax1.boxplot(strategy_accuracies, labels=strategies, patch_artist=True)
        for patch, color in zip(bp['boxes'], strategy_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Sensor Strategy', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('DeepONet Accuracy by Sensor Strategy', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
                                                         
        for i, (strategy, models_list) in enumerate(strategy_data.items()):
            sensors = [model['n_sensors'] for model in models_list]
            accuracies = [model['accuracy'] for model in models_list]
            color = self.colors[i % len(self.colors)]
            
            ax2.scatter(sensors, accuracies, c=[color]*len(sensors), 
                       label=strategy, s=120, alpha=0.7, edgecolors='black')
            
                              
            for model in models_list:
                ax2.annotate(model['name'], (model['n_sensors'], model['accuracy']), 
                           xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Number of Sensors', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('DeepONet: Sensors vs Accuracy by Strategy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 DeepONet sensor strategy analysis saved: {save_path}")
        
        return fig

    def create_summary_table(self, results_data, save_path=None):
        models = results_data.get('models', [])
        
        data = []
        for model in models:
            arch = model['model_info'].get('architecture', {})
            data.append({
                'Model': model['name'].replace('_64x64', '').replace('DeepONet', 'DON'),
                'Accuracy (%)': f"{model['metrics']['accuracy']:.2f}",
                'Relative L2': f"{model['metrics']['relative_l2']:.4f}",
                'Parameters': f"{model['model_info']['parameters']:,}",
                'Sensors': f"{arch.get('n_sensors', 'N/A')}",
                'Hidden Size': f"{arch.get('hidden_size', 'N/A')}",
                'Layers': f"{arch.get('num_layers', 'N/A')}",
                'Sensor Strategy': f"{arch.get('sensor_strategy', 'N/A')}",
                'Activation': f"{arch.get('activation', 'N/A')}",
                'Training Time (min)': f"{model['metrics'].get('wall_sec', 0)/60:.1f}",
                'MAE': f"{model['metrics']['mae']:.6f}",
                'MSE': f"{model['metrics']['mse']:.6f}"
            })
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df


def create_deeponet_charts(json_file_path, output_dir='./deeponet_charts', show_plots=False, create_individual_accuracy=True):
                             
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
                              
    chart_creator = DeepONetChartCreator()
    
                  
    try:
        results = chart_creator.load_results(json_file_path)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    
    created_files = {}
    
                                             
    curves_path = output_path / "training_curves.png"
    fig1 = chart_creator.create_training_curves(results, save_path=curves_path, create_individual=create_individual_accuracy)
    if fig1:
        created_files['training_curves'] = curves_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig1)
    
                                               
    individual_charts = chart_creator.create_individual_charts(results, output_path)
    created_files.update(individual_charts)
    
                                                   
    comparison_path = output_path / "full_model_comparison.png"
    fig2 = chart_creator.create_model_comparison(results, save_path=comparison_path)
    if fig2:
        created_files['full_model_comparison'] = comparison_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig2)
    
                                                   
    arch_path = output_path / "architecture_analysis.png"
    fig3 = chart_creator.create_architecture_analysis(results, save_path=arch_path)
    if fig3:
        created_files['architecture_analysis'] = arch_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig3)
    
                                                     
    sensor_path = output_path / "sensor_strategy_analysis.png"
    fig4 = chart_creator.create_sensor_strategy_analysis(results, save_path=sensor_path)
    if fig4:
        created_files['sensor_strategy_analysis'] = sensor_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig4)
    
                                              
    arch_space_path = output_path / "architecture_space.png"
    fig5 = chart_creator.create_architecture_space_chart(results, save_path=arch_space_path)
    if fig5:
        created_files['architecture_space'] = arch_space_path
        if show_plots:
            plt.show()
        else:
            plt.close(fig5)
    
                                           
    table_path = output_path / "summary_table.csv"
    summary_df = chart_creator.create_summary_table(results, save_path=table_path)
    if summary_df is not None:
        created_files['summary_table'] = table_path
    
                           
    models = results.get('models', [])
    if models:
        best_model = max(models, key=lambda x: x['metrics']['accuracy'])
        print(f"\n🏆 Best DeepONet Model: {best_model['name'].replace('_64x64', '')}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.2f}%")
        print(f"   Parameters: {best_model['model_info']['parameters']:,}")
        arch = best_model['model_info'].get('architecture', {})
        print(f"   Sensors: {arch.get('n_sensors', 'N/A')}")
        print(f"   Hidden Size: {arch.get('hidden_size', 'N/A')}")
        print(f"   Strategy: {arch.get('sensor_strategy', 'N/A')}")
        print(f"   Training Time: {best_model['metrics'].get('wall_sec', 0)/60:.1f} min")
    
    print(f"\nDeepONet charts created successfully!")
    
    return created_files


def create_charts_from_latest_deeponet_results(results_dir='.', output_dir='./deeponet_charts'):
                                       
    pattern = os.path.join(results_dir, 'results_deeponet_*.json')
    results_files = glob.glob(pattern)
    
                                          
    benchmark_file = os.path.join(results_dir, 'benchmark_results.json')
    if os.path.exists(benchmark_file):
        results_files.append(benchmark_file)
    
    if not results_files:
        print(f" No DeepONet results files found in {results_dir}")
        print("   Looking for files matching: results_deeponet_*.json or benchmark_results.json")
        return None
    
                              
    latest_file = max(results_files, key=os.path.getctime)
    print(f"📊 Using most recent DeepONet results: {os.path.basename(latest_file)}")
    
    return create_deeponet_charts(latest_file, output_dir)


def analyze_deeponet_results(json_file_path, output_dir='./deeponet_analysis'):
    chart_creator = DeepONetChartCreator(figsize=(14, 8))
    
                             
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
                              
    try:
        results = chart_creator.load_results(json_file_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None
    
    print(f"Analyzing DeepONet results from: {json_file_path}")
    
                       
    created_files = create_deeponet_charts(json_file_path, output_dir, create_individual_accuracy=True)
    
                          
    summary_df = chart_creator.create_summary_table(results)
    
    if summary_df is not None:
        print(f"\nDeepONet Quick Analysis:")
        print("="*80)
        display_cols = ['Model', 'Accuracy (%)', 'Sensors', 'Hidden Size', 'Parameters', 'Training Time (min)']
        available_cols = [col for col in display_cols if col in summary_df.columns]
        print(summary_df[available_cols].to_string(index=False))
    
    return chart_creator, output_path, summary_df


def quick_deeponet_analysis(json_file_path='./results_deeponet_*.json', output_dir='./deeponet_charts'):
    import glob
    
                                               
    if '*' in json_file_path:
        files = glob.glob(json_file_path)
        if not files:
            print(f"No files found matching: {json_file_path}")
            return None
        json_file_path = max(files, key=os.path.getctime)
        print(f"Using latest DeepONet results: {os.path.basename(json_file_path)}")
    
                   
    created_files = create_deeponet_charts(json_file_path, output_dir, create_individual_accuracy=True)
    
    if created_files:
        print(f"\nDeepONet charts created successfully in: {output_dir}")
        print(f"Structure matches your FNO charts format!")
        
                                
        key_files = ['training_curves', 'full_model_comparison', 'architecture_analysis', 'model_efficiency']
        for key in key_files:
            if key in created_files:
                print(f"   {key}: {created_files[key].name}")
        
                                
        individual_dir = Path(output_dir) / 'individual_accuracy_plots'
        if individual_dir.exists():
            individual_files = list(individual_dir.glob('*.png'))
            print(f"   📈 Individual plots: {len(individual_files)} files in individual_accuracy_plots/")
    
    return created_files


if __name__ == "__main__":
    print("  quick_deeponet_analysis()  # This will create the same structure as your FNO!")

quick_deeponet_analysis('./visualizer/benchmark_results.json')