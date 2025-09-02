system DecentralizedSystem { 
    components {
        GPS [SE=100];
        Boat1 [SE=100];
        Boat2 [SE=100];
        Boat3 [SE=100];
        Boat4 [SE=100];
        Boat5 [SE=100]; 
    
    dependencies {
        Boat1 -> Boat2 [alpha=0.5, beta=50];
        Boat1 -> Boat3 [alpha=1, beta=0];
        Boat4 -> Boat1 [alpha=1, beta=0];
        Boat4 -> Boat5 [alpha=1, beta=0];
        Boat3 -> Boat2 [alpha=0.5, beta=50];
    }
}

context C0 { 
    hazard GPSJamming {
       target=[GPS], 
       impact=0.7
    }
    performance {
        Function 'default'
    } 
}

resilience R0 { 
    hazards GPSJamming {
       target=[GPS], 
       impact=0.7
    }
    performances {
        Function 'default'
    } 
}
