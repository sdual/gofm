package gofm

import "gonum.org/v1/gonum/mat"

type (
	FactorizationMachine struct {
		latentDim int64
	}

	FMOption func(*FactorizationMachine)
)

func NewFactorizationMachine(options ...FMOption) FactorizationMachine {
	fm := FactorizationMachine{}
	for _, opt := range options {
		opt(&fm)
	}
	return fm
}

func WithLatentDim(dim int64) FMOption {
	return func(fm *FactorizationMachine) {
		fm.latentDim = dim
	}
}

func (fm FactorizationMachine) Predict(features mat.Matrix) mat.Vector {
	return nil
}

func (fm FactorizationMachine) Train() FMParams {
	return FMParams{}
}
