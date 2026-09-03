# Model specification

Let `A^(k)` be the training-positive association matrix in fold `k`. Static
drug and microbe similarities are fused with GIP kernels derived only from
`A^(k)`. The fused matrices define normalized top-20 propagation operators
`G_d` and `G_m`.

Four frozen biomedical language models produce standardized semantic views.
For each entity type, the semantic representation is concatenated with its
fused similarity row and training association profile. Type-specific linear
layers project these inputs to hidden dimension `H=320`.

For entity features `X` and propagation operator `G`:

```text
h0 = ReLU(WX)
h1 = G h0
h2 = G h1
O  = BiLSTM([h0, h1, h2])
alpha = softmax(Attention(O))
h = LayerNorm(Dropout(sum_r alpha_r O_r) + h0)
```

Drug and microbe encoders have separate input projections and share the
BiLSTM-attention aggregation architecture. For candidate pair `(d,m)`:

```text
z_dm = concat(h_d, h_m, h_d * h_m, abs(h_d - h_m))
score_dm = MLP(z_dm)
p_dm = sigmoid(score_dm)
```

The MLP dimensions are `1280 -> 320 -> 160 -> 1`. Training minimizes weighted
binary cross-entropy with a fold-specific positive-class weight computed from
the training labels.
