function K_prime = k_motion(K, Z,C, par)

    K_prime = par.R * K + Z * par.w - C ;

end 