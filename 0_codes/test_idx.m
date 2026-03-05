gri.z = [1; 2; 3; 4; 5];
r_grid = [0.1; 0.2; 0.3; 0.4; 0.5];
z_kron = kron(gri.z, ones(5, 1));
r_kron = kron(ones(5, 1), r_grid);
for i = 21:25
    fprintf('Index %d: z=%.1f, r=%.1f\n', i, z_kron(i), r_kron(i));
end
