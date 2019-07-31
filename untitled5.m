ind = randi([1 90],1,10);
CLASS = [];
for i = 1:10
    if (0<ind(i)) && (ind(i)<31)
        class = 1;
    elseif (30<ind(i)) && (ind(i)<61)
        class = 2;
    elseif (61<ind(i)) && (ind(i)<91)
        class = 3;
    end
    CLASS = [CLASS class];
end
