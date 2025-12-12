% Animate the Three-Link Walker
function animateFiveLink(tData, qData)
    figure(1000)

    for i =1:length(tData) 
        clf ;
        drawFiveLink(qData(i, :)');
        line([-1, 20],[0;0],'Color', 'k', 'LineWidth', 2)
        axis([-1 15 -1 5]) ;  axis equal
        grid on ;
        drawnow ;
        pause(0.001) ;
    end
end


% Draw one frame of the Three-Link Walker
function drawFiveLink(q)
    x = q(1);
    y = q(2);
    q1 = q(3);
    q2 = q(4);
    q3 = q(5);
    q4 = q(6);
    q5 = q(7);

    pH = p_H_auto([q;zeros(7,1)]);
    pS = p_S_auto([q;zeros(7,1)]);
    pSw = p_sw_auto([q;zeros(7,1)]);
    pSt = p_st_auto([q;zeros(7,1)]);
    pA3 = p_A3_auto([q;zeros(7,1)]);
    pA4 = p_A4_auto([q;zeros(7,1)]);
    
    l1 = line([x;pS(1)], [y;pS(2)], 'Color', 'k', 'LineWidth', 2);
    hold on
    l2 = line([x;pSw(1)], [y;pSw(2)], 'Color', 'b', 'LineWidth', 2);
    l3 = line([x;pSt(1)], [y;pSt(2)], 'Color', 'r', 'LineWidth', 2);
    l4 = line([pS(1);pA3(1)], [pS(2);pA3(2)], 'Color', 'b', 'LineWidth', 2);
    l5 = line([pS(1);pA4(1)], [pS(2);pA4(2)], 'Color', 'r', 'LineWidth', 2);

    plot(pSw(1), pSw(2), 'bo', 'MarkerSize',7,'MarkerEdgeColor','b','MarkerFaceColor','g')
    plot(pSt(1), pSt(2), 'ro', 'MarkerSize',7,'MarkerEdgeColor','r','MarkerFaceColor','g')
    plot(pA3(1), pA3(2), 'bo', 'MarkerSize',7,'MarkerEdgeColor','b','MarkerFaceColor','g')
    plot(pA4(1), pA4(2), 'ro', 'MarkerSize',7,'MarkerEdgeColor','r','MarkerFaceColor','g')
    
    plot(pS(1), pS(2), 'ko', 'MarkerSize',7,'MarkerEdgeColor','k','MarkerFaceColor','g')
    plot(x, y, 'ko', 'MarkerSize',7,'MarkerEdgeColor','k','MarkerFaceColor','g')
end