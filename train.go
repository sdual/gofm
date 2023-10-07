package gofm

import "gonum.org/v1/gonum/mat"

type FMLeaner struct {
	gd          GradientDescent
	hyperparams FMHyperparams
	Params      FMParams
}

func NewFMLeaner(gd GradientDescent) FMLeaner {
	return FMLeaner{
		gd: gd,
	}
}

func (fm FMLeaner) grad(paramIndex int, features mat.Vector) float64 {
	if fm.Params.IsLinear(paramIndex) {

	} else {

	}
	return 0.0
}

func (fm FMLeaner) linearGrad(paramIndex int, features mat.Vector) float64 {
	// derivative of linear terms (Σ_i w_i x_i) with respect to w_i.
	return features.AtVec(paramIndex)
}

func (fm FMLeaner) interactGrad(paramIndex int, features mat.Vector) float64 {
	// derivative of interaction terms (Σ_{i,j} <v_i, v_j> x_i x_j) with respect to the vector element (v_i)_k.
	fm.Params.At(paramIndex)
	return 0.0
}
