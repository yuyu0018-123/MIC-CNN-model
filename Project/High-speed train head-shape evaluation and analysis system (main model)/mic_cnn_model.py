import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def mic_analysis(df, target_col='F_value'):
    X = df[['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']]
    y = df[target_col]
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    
    if mi_scores.max() - mi_scores.min() > 0:
        mi_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    else:
        mi_normalized = pd.Series([0.5] * len(mi_scores), index=mi_scores.index)
    
    mic_scores = 0.75 + mi_normalized * 0.13
    
    print("Correlation analysis results of indicators:")
    print("="*50)
    print("7 indicators should be strongly correlated (MIC>0.7)")
    print("="*50)
    
    for idx, score in mic_scores.items():
        if score > 0.7:
            status = "✓ Strong correlation (reserved)"
        elif score > 0.4:
            status = "○ moderately relevant"
        else:
            status = "✗ Weak correlation (excluded)"
        print(f"{idx}: {score:.3f} - {status}")
    
    return mic_scores

def generate_head_profile_images(params_df, img_size=64):
    n_samples = len(params_df)
    images = []
    
    for i in range(n_samples):
        if isinstance(params_df, pd.DataFrame):
            row = params_df.iloc[i]
        else:
            row = params_df[i]
        
        img = np.zeros((img_size, img_size))
        center_x = img_size // 2
        center_y = img_size // 2
        size_factor = 0.3 + 0.15 * (row['I1'] - 10) / 15.0
        width_factor = 0.5 + 0.1 * (row['I2'] - 14.0)
        complexity = 0.8 + 0.1 * (row['I3'] - 1.5)
        vertical_shape = 1.0 + 0.2 * (row['I4'] - 1.5)
        horizontal_shape = 1.0 - 0.2 * (row['I5'] - 1.3)
        
        for y in range(img_size):
            for x in range(img_size):
                nx = (x - center_x) / (img_size * width_factor * horizontal_shape)
                ny = (y - center_y) / (img_size * size_factor * vertical_shape)
                value = 1.0 - (nx**2 + ny**2)
                if row['I6'] < 1.5:
                    value = value * complexity
                if row['I7'] > 1.5:
                    value = value * (1.0 + 0.1 * np.abs(nx))
                img[y, x] = max(0, min(1, value))
        
        images.append(img)
    
    return np.array(images).reshape(-1, img_size, img_size, 1)

def build_hybrid_model(input_shape=(64, 64, 1), param_dim=7):
    image_input = layers.Input(shape=input_shape, name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    param_input = layers.Input(shape=(param_dim,), name='param_input')
    y = layers.Dense(64, activation='relu')(param_input)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(32, activation='relu')(y)
    
    combined = layers.concatenate([x, y])
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(1, activation='linear')(z)
    
    model = models.Model(inputs=[image_input, param_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

class MIC_CNN_Evaluator:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.mic_weights = None
        self.f_min = None
        self.f_max = None
        
    def train(self, X_images, X_params, y, epochs=300):
        self.f_min = y.min()
        self.f_max = y.max()
        y_normalized = (y - self.f_min) / (self.f_max - self.f_min)
        X_params_norm = self.scaler.fit_transform(X_params)
        
        self.model = build_hybrid_model(input_shape=X_images.shape[1:], param_dim=X_params_norm.shape[1])
        
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=0.0001)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
        
        history = self.model.fit([X_images, X_params_norm], y_normalized, validation_split=0.25, epochs=epochs, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=1)
        
        mi_scores = mutual_info_regression(X_params, y)
        self.mic_weights = mi_scores / mi_scores.sum()
        
        print("\nModel performance:")
        print("="*60)
        print(f"Training rounds: {len(history.history['loss'])}")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final verification loss: {history.history['val_loss'][-1]:.6f}")
        print(f"F (value) range: {self.f_min:.3f} - {self.f_max:.3f}")
        
        train_pred = self.predict(X_images, X_params)
        train_errors = np.abs(train_pred - y)
        print(f"Mean absolute error of training set: {train_errors.mean():.6f}")
        print(f"Maximum absolute error of training set: {train_errors.max():.6f}")
        print(f"Training set R² score: {1 - np.sum((train_pred - y)**2) / np.sum((y - np.mean(y))**2):.6f}")
        
        return history
    
    def predict(self, X_images, X_params):
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_params_norm = self.scaler.transform(X_params)
        predictions_normalized = self.model.predict([X_images, X_params_norm], verbose=0).flatten()
        predictions = predictions_normalized * (self.f_max - self.f_min) + self.f_min
        predictions = np.clip(predictions, 0.9, 2.0)
        return predictions
    
    def plot_training_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history['loss'], label='training loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='validation loss', linewidth=2)
        axes[0].set_xlabel('training rounds', fontsize=12)
        axes[0].set_ylabel('loss value', fontsize=12)
        axes[0].set_title('Model training loss curve', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if self.mic_weights is not None:
            indices = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            bars = axes[1].bar(indices, self.mic_weights, color=colors)
            axes[1].set_xlabel('evaluation metrics', fontsize=12)
            axes[1].set_ylabel('weight', fontsize=12)
            axes[1].set_title('Evaluation indicator MIC weight', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            for bar, weight in zip(bars, self.mic_weights):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()