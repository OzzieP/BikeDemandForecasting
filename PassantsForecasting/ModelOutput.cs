using System;
using System.Collections.Generic;
using System.Text;

namespace PassantsForecasting
{
    public class ModelOutput
    {
        public float[] ForecastedPassants { get; set; }

        public float[] LowerBoundPassants { get; set; }

        public float[] UpperBoundPassants { get; set; }
    }
}
