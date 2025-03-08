import tensorflow as tf
from keras.layers import Input, Dense, LayerNormalization, Dropout, Reshape, Conv3D, Conv3DTranspose
from keras.models import Model

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def separate_heads(self, x, batch_size):
        # Ajuste para separar as cabeças corretamente no 3D
        # Forma do tensor x: (batch_size, height, width, depth, embed_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))  # Cria a divisão para as cabeças
        return tf.transpose(x, perm=[0, 2, 1, 3])  # Transpor para (batch_size, num_heads, height, width, depth, projection_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h, w, d = tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]  # Pegando dimensões 3D
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention_output = tf.matmul(query, key, transpose_b=True)
        attention_output = tf.nn.softmax(attention_output)
        attention_output = tf.matmul(attention_output, value)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, h, w, d, self.embed_dim))  # Mantendo a forma 3D
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Camadas do Transformer Block
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'), 
            Dense(embed_dim)
        ])
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # inputs e attn_output têm a mesma forma agora
        out1 = self.layernorm1(inputs + attn_output)  
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_3d(input_size):
    inputs = Input(input_size)
    
    # Encoder (Downsampling)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(inputs)  # Strides=2 para reduzir
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)  # Strides=2 para reduzir
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)  # Strides=2 para reduzir
    
    # Decoder (Upsampling)
    x = Conv3DTranspose(128, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)  # Upsample
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)  # Refining
    x = Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)  # Upsample
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)  # Refining
    x = Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)  # Upsample
    
    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, outputs)
    return model
