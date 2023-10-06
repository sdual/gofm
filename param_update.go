package gofm

type (
	GradientDescent interface {
		Update(grad float64, param float64) float64
	}

	StandardGradientDescent struct {
		lr float64
	}
)

func NewStandardGradientDescent(lr float64) StandardGradientDescent {
	return StandardGradientDescent{
		lr: lr,
	}
}

func (gd StandardGradientDescent) Update(grad float64, param float64) float64 {
	return param - gd.lr*grad
}
