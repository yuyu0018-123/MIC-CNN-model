import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from train_set import generate_train_head_samples
from mic_cnn_model import mic_analysis, generate_head_profile_images, MIC_CNN_Evaluator

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def get_user_input():
    print("\n" + "="*60)
    print("Please enter the parameters of the high-speed train model")
    print("="*60)
    
    print("\nPlease enter the number of train models:")
    n_models = int(input("Number of train models: ").strip())
    models = []
    params_list = []
    
    for i in range(n_models):
        print(f"\n--- train models {i+1} ---")
        model_name = input(f"train model name: ").strip()
        
        print("Please enter the values of 7 evaluation indicators:")
        print("(Reference range: I1:4-25, I2:13-15, I3:1-2.5, I4:1-2, I5:1-2, I6:1-2, I7:1-2)")
        
        i1 = float(input("I1 (Slenderness ratio): ").strip())
        i2 = float(input("I2 (Head cross-sectional area, m²): ").strip())
        i3 = float(input("I3 (Cross-sectional area change rate): ").strip())
        i4 = float(input("I4 (Vertical section shape coefficient): ").strip())
        i5 = float(input("I5 (Cross section shape coefficient): ").strip())
        i6 = float(input("I6 (Drag coefficient): ").strip())
        i7 = float(input("I7 (Nose taper): ").strip())
        
        models.append(model_name)
        params_list.append([i1, i2, i3, i4, i5, i6, i7])
    
    data = {
        'Model': models,
        'I1': [p[0] for p in params_list],
        'I2': [p[1] for p in params_list],
        'I3': [p[2] for p in params_list],
        'I4': [p[3] for p in params_list],
        'I5': [p[4] for p in params_list],
        'I6': [p[5] for p in params_list],
        'I7': [p[6] for p in params_list]
    }
    
    df = pd.DataFrame(data)
    print("\nThe high-speed train model parameters you entered:")
    print(df.to_string(index=False))
    
    return df

def display_evaluation_results(evaluator, df_input):
    input_images = generate_head_profile_images(df_input[['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']], img_size=64)
    input_params = df_input[['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']].values
    predictions = evaluator.predict(input_images, input_params)
    
    print("\nResults of the object to be evaluated:")
    print("="*60)
    
    print(f"{'Train model':<15} {'F(value)':<15}")
    print("-"*60)
    
    for i, model in enumerate(df_input['Model']):
        eval_val = predictions[i]
        print(f"{model:<15} {eval_val:<15.3f}")
    
    plot_evaluation_bar_chart(df_input['Model'].tolist(), predictions)
    
    return predictions

def plot_evaluation_bar_chart(models, values):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_pos = np.arange(len(models))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = axes[0].bar(x_pos, values, color=colors)
    axes[0].set_xlabel('Train model', fontsize=12)
    axes[0].set_ylabel('F(value)', fontsize=12)
    axes[0].set_title('Head shape evaluation results', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
    axes[0].axhline(y=1.8, color='red', linestyle='--', alpha=0.5)
    
    axes[1].plot(x_pos, values, 'bo-', linewidth=2, markersize=8)
    axes[1].scatter(x_pos, values, s=100, color='red', zorder=5)
    axes[1].set_xlabel('Train model', fontsize=12)
    axes[1].set_ylabel('F(value)', fontsize=12)
    axes[1].set_title('Trend chart of evaluation value', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("="*60)
    print("High speed train head type MIC-CNN evaluation system")
    print("="*60)
    
    print("\n1. 80 simulated samples for training...")
    df_samples = generate_train_head_samples(80)
    print(f"Generate training samples: {len(df_samples)}")
    
    print("\n2. Conduct correlation analysis of indicators...")
    mic_scores = mic_analysis(df_samples)
    
    print("\n3. Generate front profile images for training purposes...")
    X_images = generate_head_profile_images(df_samples[['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']], img_size=64)
    print(f"Image shape: {X_images.shape}")
    
    X_params = df_samples[['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']].values
    y = df_samples['F_value'].values
    
    print("\n4. Training MIC-CNN evaluation model...")
    print("Start training, please wait...")
    
    evaluator = MIC_CNN_Evaluator()
    history = evaluator.train(X_images, X_params, y, epochs=400)
    evaluator.plot_training_history(history)
    
    print("\n" + "="*60)
    print("Model training completed! We can now start evaluating new car models.")
    print("="*60)
    
    while True:
        df_input = get_user_input()
        predictions = display_evaluation_results(evaluator, df_input)
        
        continue_eval = input("\nDo you want to continue evaluating other car models?(y/n): ").strip().lower()
        if continue_eval != 'y':
            break
    
    print("\n" + "="*60)
    print("End of evaluation system usage!")
    print("="*60)

if __name__ == "__main__":
    main()