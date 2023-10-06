package gofm

type FMLeaner struct {
	gd          GradientDescent
	hyperParams FMHyperParams
	Params      FMParams
}

func NewFMLeaner(gd GradientDescent) FMLeaner {
	return FMLeaner{
		gd: gd,
	}
}

func (fm FMLeaner) grad(paramIndex int) float64 {
	return 0.0
}
