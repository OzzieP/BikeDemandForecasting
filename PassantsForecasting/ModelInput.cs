using System;
using System.Collections.Generic;
using System.Text;

namespace PassantsForecasting
{
    public class ModelInput
    {
        public float Semaine { get; set; }

        public float Jour { get; set; }

        public string Feu { get; set; }

        public float NbPassants { get; set; }
    }
}
