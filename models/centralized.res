system CentralizedSystem { 
    components {
        Boat1 [SE=100, criticality=0.9];
        Boat3 [SE=100, criticality=0.6];
        Boat4 [SE=100, criticality=0.5];
        Boat5 [SE=100, criticality=0.8];
        Boat6 [SE=100, criticality=0.8];
    }

    dependencies {
        Boat2 -> Boat1 [alpha=0.25, beta=75];
        BOat3 -> Boat1 [alpha=0.25, beta=75];
        Boat5 -> Boat1 [alpha=0.25, beta=75];
        Boat4 -> Boat1 [alpha=0.25, beta=75];
        Boat3 -> Boat5 [alpha=1, beta=0];
        Boat3 -> Boat2 [alpha=1, beta=0]

    }
}

context C0 { 
    hazard GPSSpoofing [target=[D1,D4], latency=10];
    performance Score [weight=0.1, goal=maximize] ;
}