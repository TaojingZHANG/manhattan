function auc = calc_auc(x, do_plot, label, show_title)
thresh = 0.25;
errors = x;
errors = sort(errors);
errors(errors > thresh) = thresh;
prec = [errors; thresh];
rec = [1:length(errors) length(errors)]./(length(errors));
auc = trapz(prec, rec)/thresh;
if do_plot
plot(prec, rec, '.-')
if show_title
    title(strcat(label, sprintf(': %2.4f', auc*100)))
end
disp(sprintf('AUC: %2.4f', auc*100))
grid on
else
    fprintf(strcat(label, sprintf(': %2.4f', auc*100),'\n'));
end
