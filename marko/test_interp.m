c_i_int = rand(2000, 20);
k_i = rand(2000, 1)*10;
gri_k = linspace(0, 10, 20);
tic;
for j = 1:2000
    c_i(j,1) = interp_kp(gri_k, c_i_int(j, :), k_i(j,1));
end
toc;
