package gofm

import "gonum.org/v1/gonum/mat"

type (
	FMLeaner struct {
		gd          GradientDescent
		hyperparams FMHyperparams
		Params      FMParams
		featureDim  int
		maxItr      int
		loss        Loss
	}

	FMLeanerOpts func(*FMLeaner)
)

func NewFMLeaner(latentDim int, featureDim int, maxItr int, lr float64, opts ...FMLeanerOpts) FMLeaner {
	fm := &FMLeaner{
		gd:         NewStandardGradientDescent(lr),
		Params:     NewFMParams(latentDim, featureDim),
		featureDim: featureDim,
		maxItr:     maxItr,
		loss:       NewBinaryCrossEntropy(),
	}

	for _, opt := range opts {
		opt(fm)
	}
	return *fm
}

func WithOptimizer(gd GradientDescent) FMLeanerOpts {
	return func(fm *FMLeaner) {
		fm.gd = gd
	}
}

func WithLossFunc(loss Loss) FMLeanerOpts {
	return func(fm *FMLeaner) {
		fm.loss = loss
	}
}

func (fm FMLeaner) Train() FMParams {
	for i := 0; i < fm.maxItr; i++ {

	}
	return FMParams{}
}

func (fm FMLeaner) updateParams(features mat.Vector) {
	for paramIndex := 0; paramIndex < fm.Params.totalLen; paramIndex++ {
		_ = fm.grad(paramIndex, features)
	}
}

func (fm FMLeaner) grad(paramIndex int, features mat.Vector) float64 {
	if fm.Params.IsLinear(paramIndex) {
		return fm.linearGrad(paramIndex, features)
	} else {
		return fm.interactGrad(paramIndex, features)
	}
}

// linearGrad calculates derivative of linear terms (Σ_i w_i x_i) with respect to w_i.
func (fm FMLeaner) linearGrad(paramIndex int, features mat.Vector) float64 {
	return features.AtVec(paramIndex)
}

// interactGrad calculates derivative of interaction terms (Σ_{i,j} <v_i, v_j> x_i x_j) with respect to the vector element (v_i)_k.
// the derivative with respect to (v_i)_k is (Σ_k (v_j)_k) x_i x_j (j ≠ k)
func (fm FMLeaner) interactGrad(paramIndex int, features mat.Vector) float64 {
	latentVecElementIndex := fm.Params.ToLatentVecElementIndex(paramIndex)
	latentVecIndex := fm.Params.ToLatentVecIndex(paramIndex)

	// derivative of interaction term with respect to (v_i)_k is Σ_j (v_j)_k x_i x_j
	// In the following code, latentVecIndex correspond to subscript i.
	derivative := 0.0
	for j := 0; j < fm.featureDim; j++ {
		if j == latentVecIndex {
			continue
		}
		//(v_j)_k x_i x_j
		derivative += fm.Params.Interact.latentVecs[j].vector.AtVec(latentVecElementIndex) * features.AtVec(latentVecIndex) * features.AtVec(j)
	}

	return derivative
}
