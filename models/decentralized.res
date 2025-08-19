system DecentralizedSystem { 
    components {
        Boat1 [SE=100, criticality=0.9];
        Boat2 [SE=100, criticality=0.45];
        Boat3 [SE=100, criticality=0.45];
        Boat4 [SE=100, criticality=0.9];
        Boat5 [SE=100, criticality=0.45];
    }

    dependencies {
        Boat1 -> Boat2 [alpha=0.5, beta=50];
        Boat1 -> Boat3 [alpha=1, beta=0];
        Boat4 -> Boat1 [alpha=1, beta=0];
        Boat4 -> Boat5 [alpha=1, beta=0];
        Boat3 -> Boat2 [alpha=0.5, beta=50];
    }
}

context C0 { 
    hazard GPSSpoofing [target=[D1,D4], latency=10];
    performance Score [weight=0.1, goal=maximize] ;
}