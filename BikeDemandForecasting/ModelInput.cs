using System;
using System.Collections.Generic;
using System.Text;

namespace BikeDemandForecasting
{
    public class ModelInput
    {
        public DateTime RentalDate { get; set; }

        public float Year { get; set; }

        public float TotalRentals { get; set; }
    }
}
